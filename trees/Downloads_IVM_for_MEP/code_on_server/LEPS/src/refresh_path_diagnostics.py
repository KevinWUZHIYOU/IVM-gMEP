#!/usr/bin/env python3
"""Refresh path-only diagnostics in saved LEPS runs.

This never calls the LEPS energy/force backend.  It updates only quantities that
are deterministic functions of the already stored path coordinates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from leps_core import atomic_save_npz, equidistance_diagnostics


def refresh_file(path: Path) -> None:
    with np.load(path, allow_pickle=False) as saved:
        payload = {key: saved[key] for key in saved.files}
    paths = np.asarray(payload["path"], dtype=float)
    diagnostics = [equidistance_diagnostics(item) for item in paths]
    for key in (
        "i_arc",
        "equidistance_integral",
        "spring_force_max_unit",
        "equidistance_relative_defect",
        "segment_length_min",
        "segment_length_mean",
        "segment_length_max",
        "segment_length_cv",
        "segment_length_ratio",
    ):
        payload[key] = np.asarray([row[key] for row in diagnostics], dtype=float)
    atomic_save_npz(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=("runs/ivm/*.npz", "runs/neb/*.npz"),
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    files = sorted(
        {candidate for pattern in args.patterns for candidate in root.glob(pattern)}
    )
    for file_path in files:
        refresh_file(file_path)
    print(f"refreshed {len(files)} files")


if __name__ == "__main__":
    main()
