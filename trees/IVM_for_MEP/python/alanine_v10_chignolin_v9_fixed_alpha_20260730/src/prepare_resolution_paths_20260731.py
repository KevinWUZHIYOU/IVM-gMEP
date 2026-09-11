#!/usr/bin/env python3
"""Resample the common Ala initial polyline at declared image resolutions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ala_core import array_sha256, atomic_save_npz


def resample_polyline(path: np.ndarray, total_images: int) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[0] < 3:
        raise ValueError(f"invalid source path shape: {path.shape}")
    if total_images < 3:
        raise ValueError("total_images must include at least one internal image")
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    if np.any(segment_lengths <= 0.0):
        raise ValueError("source path contains a zero-length segment")
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    targets = np.linspace(0.0, float(cumulative[-1]), total_images)
    output = np.empty((total_images, path.shape[1]), dtype=float)
    for index, target in enumerate(targets):
        if index == 0:
            output[index] = path[0]
            continue
        if index == total_images - 1:
            output[index] = path[-1]
            continue
        segment = int(np.searchsorted(cumulative, target, side="right") - 1)
        segment = min(segment, path.shape[0] - 2)
        fraction = (
            (target - cumulative[segment]) / segment_lengths[segment]
        )
        output[index] = (
            (1.0 - fraction) * path[segment]
            + fraction * path[segment + 1]
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--source",
        default="inputs/common_initial_path_equidistant.npz",
    )
    parser.add_argument("--total-images", nargs="+", type=int, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    source = Path(args.source).expanduser()
    if not source.is_absolute():
        source = root / source
    with np.load(source, allow_pickle=False) as saved:
        source_path = np.asarray(saved["initial_path"], dtype=float)
    for total_images in args.total_images:
        output_path = resample_polyline(source_path, total_images)
        destination = (
            root / "inputs"
            / f"common_initial_path_equidistant_n{total_images}.npz"
        )
        atomic_save_npz(
            destination,
            initial_path=output_path,
            source_path_file=np.array(str(source)),
            source_path_sha256=np.array(array_sha256(source_path)),
            initial_path_sha256=np.array(array_sha256(output_path)),
            construction=np.array(
                "equal-arc-length resampling of the source piecewise-linear path"
            ),
            total_images=np.array(total_images),
            internal_images=np.array(total_images - 2),
        )
        print(
            f"saved {destination} shape={output_path.shape} "
            f"sha256={array_sha256(output_path)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
