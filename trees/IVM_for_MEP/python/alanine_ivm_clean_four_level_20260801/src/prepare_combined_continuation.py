#!/usr/bin/env python3
"""Extract one exactly stored combined-run macrostate as an audited continuation."""

from __future__ import annotations

import argparse
import hashlib
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--force-evaluations", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = args.result.expanduser().resolve()
    output = args.output.expanduser().resolve()
    with np.load(result, allow_pickle=False) as saved:
        labels = np.asarray(saved["stored_force_evaluations"], dtype=int)
        matches = np.flatnonzero(labels == args.force_evaluations)
        if not matches.size:
            raise ValueError(
                f"Force label {args.force_evaluations} is not stored in {result}"
            )
        index = int(matches[-1])
        path = np.asarray(saved["stored_paths"][index], dtype=float)
        history_matches = np.flatnonzero(
            np.asarray(saved["total_force_evaluations"], dtype=int)
            == args.force_evaluations
        )
        if not history_matches.size:
            raise ValueError("Selected stored path has no matching published history state")
        history_index = int(history_matches[-1])
        i_perp = float(saved["i_perp"][history_index])
        common_initial_sha256 = str(
            np.asarray(saved["common_initial_path_sha256"]).item()
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        initial_path=path,
        path_sha256=np.array(array_sha256(path)),
        source_result=np.array(str(result)),
        source_result_sha256=np.array(file_sha256(result)),
        source_force_evaluations=np.array(args.force_evaluations),
        source_i_perp=np.array(i_perp),
        source_common_initial_path_sha256=np.array(common_initial_sha256),
        continuation_policy=np.array(
            "exact stored complete macrostate; fresh optimizer memory; "
            "continuation initialization remains charged"
        ),
    )
    print(output)


if __name__ == "__main__":
    main()
