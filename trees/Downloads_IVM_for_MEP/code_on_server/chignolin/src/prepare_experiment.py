#!/usr/bin/env python3
"""Create and audit the one common central-difference IDPP path."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chignolin_core import (
    N_INTERNAL,
    TANGENT_DEFINITION,
    array_sha256,
    atomic_save_npz,
    build_central_idpp_path,
    central_tangents,
    load_endpoints,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    input_dir = root / "inputs"
    output = root / "inputs" / "common_initial_path.npz"

    path = build_central_idpp_path(input_dir, n_internal=N_INTERNAL)
    template, start, end = load_endpoints(input_dir)
    tangent = central_tangents(path)
    segments = np.linalg.norm(path[1:] - path[:-1], axis=1)
    digest = array_sha256(path)

    atomic_save_npz(
        output,
        initial_path=path,
        path_sha256=np.array(digest),
        tangent_definition=np.array(TANGENT_DEFINITION),
        n_internal=np.array(N_INTERNAL),
        endpoint_start=start,
        endpoint_end=end,
        atomic_numbers=template.get_atomic_numbers(),
        segment_lengths=segments,
    )
    report = {
        "path_file": str(output),
        "path_shape": list(path.shape),
        "path_sha256": digest,
        "n_internal": N_INTERNAL,
        "tangent_definition": TANGENT_DEFINITION,
        "tangent_norm_min": float(np.min(np.linalg.norm(tangent, axis=1))),
        "tangent_norm_max": float(np.max(np.linalg.norm(tangent, axis=1))),
        "start_exact": bool(np.array_equal(path[0], start)),
        "end_exact": bool(np.array_equal(path[-1], end)),
        "segment_length_min_A": float(np.min(segments)),
        "segment_length_mean_A": float(np.mean(segments)),
        "segment_length_max_A": float(np.max(segments)),
        "segment_length_ratio": float(np.max(segments) / np.min(segments)),
    }
    write_json(root / "reports" / "initialization_audit.json", report)
    print(report)


if __name__ == "__main__":
    main()

