#!/usr/bin/env python3
"""Run a registered list of Chignolin IR-IVM settings sequentially."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

from run_chignolin_ir_ivm_member import member_label


def parse_setting(text: str) -> Tuple[float, float, int]:
    fields = text.split(",")
    if len(fields) != 3:
        raise ValueError("each setting must be alpha,r,m_max")
    return float(fields[0]), float(fields[1]), int(fields[2])


def atomic_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def result_glob(root: Path, tier: str, label: str) -> List[Path]:
    return sorted((root / "runs" / tier).glob(f"ivm_{label}_*.npz"))


def run_batch(args: argparse.Namespace) -> None:
    root = Path(args.root).expanduser().resolve()
    settings = [parse_setting(item) for item in args.settings]
    lock_path = root / "logs" / "chignolin_ir_ivm_host.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another Chignolin IR-IVM batch owns {lock_path}"
            ) from error
        status_path = root / "reports" / "campaign_status" / f"{args.name}.json"
        records: List[Dict[str, object]] = []
        started = time.time()
        for index, (alpha, ratio, cap) in enumerate(settings):
            label = member_label(alpha, ratio, cap, args.internal_nf_budget)
            existing = result_glob(root, args.tier, label)
            record: Dict[str, object] = {
                "index": index,
                "alpha": alpha,
                "r": ratio,
                "m_max": cap,
                "label": label,
            }
            if len(existing) == 1:
                record.update(status="already_complete", result=str(existing[0]))
                records.append(record)
                atomic_json(
                    status_path,
                    {
                        "name": args.name,
                        "started_unix": started,
                        "updated_unix": time.time(),
                        "records": records,
                    },
                )
                continue
            if len(existing) > 1:
                raise RuntimeError(f"multiple results match {label}")
            command = [
                sys.executable,
                str(root / "src" / "run_chignolin_ir_ivm_member.py"),
                "--root",
                str(root),
                "--alpha",
                f"{alpha:g}",
                "--residual-ratio",
                f"{ratio:g}",
                "--inner-max-steps",
                str(cap),
                "--internal-nf-budget",
                str(args.internal_nf_budget),
                "--tier",
                args.tier,
            ]
            record.update(status="running", command=command, started_unix=time.time())
            records.append(record)
            atomic_json(
                status_path,
                {
                    "name": args.name,
                    "started_unix": started,
                    "updated_unix": time.time(),
                    "records": records,
                },
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(root / "src")
            completed = subprocess.run(
                command,
                cwd=str(root),
                env=environment,
                check=False,
            )
            record["returncode"] = int(completed.returncode)
            record["finished_unix"] = time.time()
            if completed.returncode != 0:
                record["status"] = "failed"
                atomic_json(
                    status_path,
                    {
                        "name": args.name,
                        "started_unix": started,
                        "updated_unix": time.time(),
                        "records": records,
                    },
                )
                raise RuntimeError(f"member failed: {label}")
            produced = result_glob(root, args.tier, label)
            if len(produced) != 1:
                raise RuntimeError(f"member did not produce exactly one result: {label}")
            record.update(status="complete", result=str(produced[0]))
            atomic_json(
                status_path,
                {
                    "name": args.name,
                    "started_unix": started,
                    "updated_unix": time.time(),
                    "records": records,
                },
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default=str(Path(__file__).resolve().parents[1])
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--tier", choices=("smoke", "screen", "production"), required=True)
    parser.add_argument("--internal-nf-budget", type=int, required=True)
    parser.add_argument("--settings", nargs="+", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_batch(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
