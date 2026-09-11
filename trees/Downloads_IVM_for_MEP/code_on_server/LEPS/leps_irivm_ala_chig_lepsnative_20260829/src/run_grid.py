#!/usr/bin/env python3
"""Serial, host-locked runner for the preregistered LEPS IR-IVM grid."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import tempfile
import time

import numpy as np

from leps_irivm_lepsnative import (
    CAMPAIGN_ROOT,
    Controls,
    GRID,
    run_one,
    validate_preregistration,
)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, path)


def existing_result(control):
    path = CAMPAIGN_ROOT / "runs" / f"{control.run_id}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as saved:
        config = json.loads(str(saved["config_json"].item()))
        if config != control.as_dict():
            raise RuntimeError(f"existing result/control mismatch: {path}")
        return {
            "result": str(path),
            "terminal_article_nf": int(saved["terminal_article_nf"].item()),
            "stop_reason": str(saved["stop_reason"].item()),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.jobs != 1:
        raise RuntimeError("formal LEPS campaign is serial-only: jobs must be 1")
    validate_preregistration()
    controls = [Controls(*triple) for triple in GRID]
    records = [
        {
            "index": index,
            "controls": control.as_dict(),
            "run_id": control.run_id,
            "status": "pending",
        }
        for index, control in enumerate(controls)
    ]
    status_path = CAMPAIGN_ROOT / "reports/campaign_status.json"
    started = time.time()

    def document():
        states = [record["status"] for record in records]
        return {
            "campaign": "LEPS-native Ala/Chignolin-style IR-IVM clean grid",
            "host": socket.gethostname(),
            "jobs": 1,
            "parameter_combination_count": len(records),
            "complete_count": sum(state == "complete" for state in states),
            "running_count": sum(state == "running" for state in states),
            "failed_count": sum(state == "failed" for state in states),
            "pending_count": sum(state == "pending" for state in states),
            "cumulative_terminal_article_nf": sum(
                int(record.get("terminal_article_nf", 0)) for record in records
            ),
            "started_unix": started,
            "updated_unix": time.time(),
            "records": records,
        }

    locks = CAMPAIGN_ROOT / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / "host-local-leps-grid.lock").open("a+") as batch_handle:
        fcntl.flock(batch_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for control, record in zip(controls, records):
            prior = existing_result(control)
            if prior is not None:
                record.update(status="complete", resumed=True, **prior)
        atomic_json(status_path, document())

        for control, record in zip(controls, records):
            if record["status"] == "complete":
                continue
            record.update(status="running", started_unix=time.time())
            atomic_json(status_path, document())
            try:
                with (locks / "host-local-leps-irivm.lock").open("a+") as method_handle:
                    fcntl.flock(
                        method_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                    destination, result = run_one(control)
                record.update(
                    status="complete",
                    result=str(destination),
                    terminal_article_nf=int(result["terminal_article_nf"]),
                    stop_reason=result["stop_reason"],
                    finished_unix=time.time(),
                )
                print(
                    f"{record['index'] + 1}/{len(records)} "
                    f"alpha={control.alpha:g} r={control.r:g} "
                    f"m_max={control.m_max} NF={result['terminal_article_nf']} "
                    f"stop={result['stop_reason']}",
                    flush=True,
                )
            except Exception as error:
                record.update(
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                    finished_unix=time.time(),
                )
                atomic_json(status_path, document())
                raise
            atomic_json(status_path, document())
    print(status_path)


if __name__ == "__main__":
    main()
