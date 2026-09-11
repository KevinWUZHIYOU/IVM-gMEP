#!/usr/bin/env python3
"""Create an auditable equal-spring continuation from a saved terminal path."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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


def load_runner(study: Path):
    runner_path = (
        study
        / "src/run_ivm_v9_spring_projected_production_20260730.py"
    )
    spec = importlib.util.spec_from_file_location("ala_ivm_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resample_polyline(path: np.ndarray, total_images: int) -> np.ndarray:
    """Place the same image count at equal arclength on the input polyline."""

    path = np.asarray(path, dtype=float)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("source path contains a zero-length segment")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, float(cumulative[-1]), int(total_images))
    output = np.empty((int(total_images), path.shape[1]), dtype=float)
    output[0] = path[0]
    output[-1] = path[-1]
    for index, target in enumerate(targets[1:-1], start=1):
        segment = int(np.searchsorted(cumulative, target, side="right") - 1)
        segment = min(segment, path.shape[0] - 2)
        fraction = (target - cumulative[segment]) / lengths[segment]
        output[index] = (
            (1.0 - fraction) * path[segment]
            + fraction * path[segment + 1]
        )
    return output


def main() -> None:
    study = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--geometry-operator",
        choices=(
            "equal_spring",
            "piecewise_linear_arclength",
            "piecewise_linear_then_equal_spring",
        ),
        default="equal_spring",
    )
    parser.add_argument(
        "--total-images",
        type=int,
        default=0,
        help="Output image count; zero preserves the source count.",
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()

    with np.load(source, allow_pickle=False) as saved:
        terminal = np.asarray(saved["final_path"], dtype=float)
        source_initial_sha = str(np.asarray(saved["initial_path_sha256"]).item())
        source_configuration_sha = str(
            np.asarray(saved["configuration_sha256"]).item()
        )
        source_final_iperp = float(np.asarray(saved["i_perp"])[-1])
        source_final_call = int(np.asarray(saved["calls_per_image"])[-1])

    runner = load_runner(study)
    if args.geometry_operator == "equal_spring":
        reparameterized, residual = runner.central_spring_force_equilibrium(
            terminal
        )
        operator_label = "central_spring_force_equilibrium"
    else:
        total_images = args.total_images or terminal.shape[0]
        if total_images < 3:
            raise ValueError("total-images must include an internal image")
        reparameterized = resample_polyline(terminal, total_images)
        residual = 0.0
        operator_label = "piecewise_linear_equal_arclength"
        if args.geometry_operator == "piecewise_linear_then_equal_spring":
            reparameterized, residual = (
                runner.central_spring_force_equilibrium(reparameterized)
            )
            operator_label = (
                "piecewise_linear_equal_arclength_then_"
                "central_spring_force_equilibrium"
            )
    before = runner.equidistance_diagnostics(terminal)
    after = runner.equidistance_diagnostics(reparameterized)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        initial_path=reparameterized,
        path_sha256=np.array(array_sha256(reparameterized)),
        source_result=np.array(str(source)),
        source_result_sha256=np.array(file_sha256(source)),
        source_initial_path_sha256=np.array(source_initial_sha),
        source_configuration_sha256=np.array(source_configuration_sha),
        source_final_iperp=np.array(source_final_iperp),
        source_final_call=np.array(source_final_call),
        geometry_operator=np.array(operator_label),
        geometry_residual=np.array(float(residual)),
        source_segment_length_cv=np.array(before["segment_length_cv"]),
        output_segment_length_cv=np.array(after["segment_length_cv"]),
        continuation_policy=np.array(
            "fresh optimizer state from a geometry-only equal-spring "
            "reparameterization of the saved terminal Cartesian path"
        ),
    )
    audit = {
        "output": str(output),
        "output_sha256": file_sha256(output),
        "path_sha256": array_sha256(reparameterized),
        "source": str(source),
        "source_sha256": file_sha256(source),
        "source_final_call": source_final_call,
        "source_final_iperp": source_final_iperp,
        "geometry_operator": operator_label,
        "geometry_residual": float(residual),
        "source_segment_length_cv": float(before["segment_length_cv"]),
        "output_segment_length_cv": float(after["segment_length_cv"]),
    }
    output.with_suffix(".json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
