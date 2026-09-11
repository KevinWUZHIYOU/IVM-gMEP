#!/usr/bin/env python3
"""Create the single common 22-image LEPS IDPP path."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from leps_core import (
    TANGENT_DEFINITION,
    array_sha256,
    atomic_save_npz,
    build_common_idpp_path,
    flat_to_physical,
    project_relative,
    reference_saddle,
    write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    path, source = build_common_idpp_path()
    output = root / "inputs" / "common_initial_path.npz"
    atomic_save_npz(
        output,
        initial_path=path,
        physical_path=flat_to_physical(path),
        source=np.asarray(source),
        tangent_definition=np.asarray(TANGENT_DEFINITION),
        sha256=np.asarray(array_sha256(path)),
    )
    saddle = reference_saddle()
    write_json(
        root / "inputs" / "initialization_audit.json",
        {
            "initial_path": project_relative(output, root),
            "source": source,
            "shape": list(path.shape),
            "sha256": array_sha256(path),
            "maximum_nonphysical_coordinate": float(
                np.max(np.abs(path[:, [1, 2, 4, 5]]))
            ),
            "segment_lengths": np.linalg.norm(
                np.diff(path, axis=0), axis=1
            ).tolist(),
            "tangent_definition": TANGENT_DEFINITION,
            "reference_saddle": {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in saddle.items()
            },
        },
    )
    print(output)


if __name__ == "__main__":
    main()
