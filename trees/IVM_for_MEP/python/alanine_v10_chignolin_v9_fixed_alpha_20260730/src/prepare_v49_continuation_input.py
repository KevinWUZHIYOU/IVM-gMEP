#!/usr/bin/env python3
"""Freeze an accepted terminal path as a restartable continuation input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    study = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=study
        / "runs/ivm/ivm_ala_v49_production_full_alpha_v9_pathsum_r0p96_"
        "stall_to_r0p95_trust1p5_stop1p5e3_a20p8_central_state_ivm_"
        "cfgd17c08618165_budget220.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=study / "inputs/v49_terminal_continuation.npz",
    )
    args = parser.parse_args()
    with np.load(args.source, allow_pickle=False) as saved:
        terminal = np.asarray(saved["final_path"], dtype=float)
        source_initial_sha = str(np.asarray(saved["initial_path_sha256"]).item())
        source_configuration_sha = str(
            np.asarray(saved["configuration_sha256"]).item()
        )
        source_final_iperp = float(np.asarray(saved["i_perp"])[-1])
        source_final_call = int(np.asarray(saved["calls_per_image"])[-1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        initial_path=terminal,
        path_sha256=np.array(array_sha256(terminal)),
        source_result=np.array(str(args.source.resolve())),
        source_result_sha256=np.array(file_sha256(args.source)),
        source_initial_path_sha256=np.array(source_initial_sha),
        source_configuration_sha256=np.array(source_configuration_sha),
        source_final_iperp=np.array(source_final_iperp),
        source_final_call=np.array(source_final_call),
        continuation_policy=np.array(
            "fresh optimizer state from the source result's accepted "
            "terminal Cartesian path"
        ),
    )
    audit = {
        "output": str(args.output.resolve()),
        "output_sha256": file_sha256(args.output),
        "path_sha256": array_sha256(terminal),
        "source": str(args.source.resolve()),
        "source_sha256": file_sha256(args.source),
        "source_final_call": source_final_call,
        "source_final_iperp": source_final_iperp,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
