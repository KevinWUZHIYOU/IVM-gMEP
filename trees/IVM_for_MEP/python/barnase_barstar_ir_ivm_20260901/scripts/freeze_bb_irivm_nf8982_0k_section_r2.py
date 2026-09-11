#!/usr/bin/env python3
"""Freeze the exact IR-IVM NF8982-centered 59x28 section coordinates.

This stage is deliberately zero-OpenMM.  It stores both the complete surface
grid and the exact 32-image source path so a later energy-only evaluator never
has to reconstruct or approximate either coordinate set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plot_bb_irivm_evolution_hashbound as common  # noqa: E402


SCHEMA = "bb-irivm-nf8982-0k-section-freeze-contract-v2"
OUTPUT_SCHEMA = "bb-irivm-nf8982-0k-section-coordinate-manifest-v2"
STATUS = "FROZEN_BEFORE_ANY_SECTION_ENERGY_CALL"
POINTS = 59
NORMAL_POINTS = 28
N_IMAGES = 32
ATOMS = 3132
DIMENSION = 3 * ATOMS
ZERO_INDEX = 4
N_GRID = np.arange(-0.12, 0.6901, 0.03, dtype=np.float64)
N_GRID[ZERO_INDEX] = 0.0


def array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def scalar(saved, key: str):
    value = np.asarray(saved[key])
    common.require(value.shape == (), f"{key} is not scalar")
    return value.item()


def arc(path: np.ndarray) -> np.ndarray:
    flat = common._as_band(path).reshape(len(path), -1)
    lengths = np.linalg.norm(np.diff(flat, axis=0), axis=1)
    common.require(
        np.all(np.isfinite(lengths)) and np.all(lengths > 0.0),
        "invalid source path segments",
    )
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return cumulative / cumulative[-1]


def resample(path: np.ndarray, target: np.ndarray) -> np.ndarray:
    band = common._as_band(path)
    flat = band.reshape(len(band), -1)
    source = arc(band)
    result = np.stack(
        [np.interp(target, source, flat[:, column]) for column in range(flat.shape[1])],
        axis=1,
    ).reshape(len(target), band.shape[1], 3)
    result[0] = band[0]
    result[-1] = band[-1]
    return np.ascontiguousarray(result, dtype=np.float64)


def basis(center: np.ndarray, initial: np.ndarray):
    flat = center.reshape(POINTS, -1)
    tangent = np.empty_like(flat)
    tangent[1:-1] = flat[2:] - flat[:-2]
    tangent[0] = flat[1] - flat[0]
    tangent[-1] = flat[-1] - flat[-2]
    tangent_norm = np.linalg.norm(tangent, axis=1)
    common.require(np.all(tangent_norm > 1.0e-14), "degenerate center tangent")
    tangent /= tangent_norm[:, None]

    displacement = initial.reshape(POINTS, -1) - flat
    displacement -= np.sum(displacement * tangent, axis=1)[:, None] * tangent
    size = np.linalg.norm(displacement, axis=1)
    valid = size > 1.0e-12
    common.require(np.count_nonzero(valid) >= 3, "insufficient normal anchors")
    normal = np.zeros_like(displacement)
    normal[valid] = displacement[valid] / size[valid, None]
    anchors = np.flatnonzero(valid)
    for index in np.flatnonzero(~valid):
        candidate = normal[anchors[np.argmin(np.abs(anchors - index))]].copy()
        candidate -= np.dot(candidate, tangent[index]) * tangent[index]
        candidate_norm = np.linalg.norm(candidate)
        common.require(candidate_norm > 1.0e-14, "degenerate borrowed normal")
        normal[index] = candidate / candidate_norm
    return tangent.reshape(center.shape), normal.reshape(center.shape)


def atomic_npz_no_overwrite(path: Path, payload: dict) -> None:
    common.require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    common.require(not temporary.exists(), f"stale temporary output: {temporary}")
    try:
        np.savez_compressed(temporary, **payload)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_and_validate_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    common.require(
        contract.get("schema") == SCHEMA and contract.get("status") == STATUS,
        "section R2 contract is not frozen",
    )
    common.require(
        contract.get("physical_execution_authorized") is False,
        "coordinate freeze contract must not authorize physical calls",
    )
    common.require(
        contract.get("geometry")
        == {
            "center_points": POINTS,
            "normal_points": NORMAL_POINTS,
            "normal_start_A": -0.12,
            "normal_stop_A": 0.69,
            "normal_step_A": 0.03,
            "zero_normal_index": ZERO_INDEX,
            "zero_normal_is_exact_float64_zero": True,
        },
        "registered geometry mismatch",
    )
    common.require(
        common.resolve_binding(path, contract["builder"], "R2 coordinate builder")
        == Path(__file__).resolve(),
        "coordinate builder path mismatch",
    )
    supersession = common.resolve_binding(
        path, contract["supersession_receipt"], "R1 supersession receipt"
    )
    receipt = json.loads(supersession.read_text(encoding="utf-8"))
    common.require(
        receipt.get("status") == "R1_SUPERSEDED_BEFORE_ANY_PHYSICAL_ENERGY_CALL"
        and int(receipt.get("physical_energy_or_force_calls_started", -1)) == 0,
        "R1 supersession is not a zero-physical-call closure",
    )
    return contract


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    contract_path = args.contract.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    contract = load_and_validate_contract(contract_path)
    common.require(
        str(output) == str(contract.get("planned_coordinate_output", "")),
        "coordinate output path differs from frozen contract",
    )
    base = common.resolve_binding(
        contract_path, contract["base_evolution_contract"], "base evolution contract"
    )
    base_contract = json.loads(base.read_text(encoding="utf-8"))
    gates = common.validate_json_gates_before_selected_npz(base, base_contract)
    common.require(int(gates["actual_nf"]) == 9985, "unexpected selected terminal NF")
    selected_result = gates["paths"]["selected_result"]
    with np.load(selected_result, allow_pickle=False) as saved:
        selected = common.collect_selected_committed_paths(
            saved, gates["verification"], int(gates["actual_nf"])
        )
    common.require(8982 in selected and 9985 not in selected, "NF8982/tail policy mismatch")

    source = selected[8982]
    source_path32 = np.ascontiguousarray(
        common._as_band(source["path"]), dtype=np.float64
    )
    common.require(source_path32.shape == (N_IMAGES, ATOMS, 3), "source path shape mismatch")
    target = np.linspace(0.0, 1.0, POINTS, dtype=np.float64)
    center = resample(source_path32, target)
    initial = resample(selected[-1]["path"], target)
    tangents, normals = basis(center, initial)
    initial_n = np.sum((initial - center) * normals, axis=(1, 2)) / np.sqrt(
        float(ATOMS)
    )
    common.require(
        float(np.min(initial_n)) >= float(N_GRID[0])
        and float(np.max(initial_n)) <= float(N_GRID[-1]),
        "registered normal grid does not cover the common initial path",
    )
    coordinates = (
        center[:, None, :, :]
        + N_GRID[None, :, None, None] * np.sqrt(float(ATOMS)) * normals[:, None, :, :]
    )
    coordinates[:, ZERO_INDEX] = center
    coordinates = np.ascontiguousarray(coordinates, dtype=np.float64)
    common.require(np.array_equal(coordinates[:, ZERO_INDEX], center), "n=0 column is not exact center")
    common.require(
        np.array_equal(center[[0, -1]], source_path32[[0, -1]]),
        "resampled center changed fixed endpoints",
    )
    common.require(np.all(np.isfinite(coordinates)), "surface coordinates are non-finite")
    hashes = np.asarray(
        [
            [array_sha(coordinates[i, j]) for j in range(NORMAL_POINTS)]
            for i in range(POINTS)
        ]
    )

    payload = {
        "schema": np.asarray(OUTPUT_SCHEMA),
        "status": np.asarray(STATUS),
        "freeze_contract": np.asarray(str(contract_path)),
        "freeze_contract_sha256": np.asarray(common.sha256_file(contract_path)),
        "builder": np.asarray(str(Path(__file__).resolve())),
        "builder_sha256": np.asarray(common.sha256_file(Path(__file__).resolve())),
        "base_evolution_contract": np.asarray(str(base)),
        "base_evolution_contract_sha256": np.asarray(common.sha256_file(base)),
        "center_source_optimization_N_F": np.asarray(8982, dtype=np.int64),
        "center_source_call_per_image": np.asarray(source["call_per_image"], dtype=np.int64),
        "terminal_partial_tail_excluded": np.asarray(True),
        "original_path32_A": source_path32,
        "original_path32_sha256": np.asarray(array_sha(source_path32)),
        "original_path32_verifier_sha256": np.asarray(
            source["verifier_dtype_shape_sha256"]
        ),
        "center_path59_A": center,
        "center_path59_sha256": np.asarray(array_sha(center)),
        "center_s": target,
        "common_initial_path59_A": initial,
        "common_initial_projected_n_A": initial_n,
        "central_tangents": tangents,
        "normal_directions": normals,
        "normal_coordinate_rms_A": N_GRID,
        "zero_normal_index": np.asarray(ZERO_INDEX, dtype=np.int64),
        "coordinates_A": coordinates,
        "coordinate_sha256": hashes,
        "coordinate_space": np.asarray("Cartesian Angstrom"),
        "surface_kind": np.asarray(
            "unrelaxed current-IR-IVM-NF8982-centered single-normal 0 K section"
        ),
        "surface_energy_only_calls_planned": np.asarray(
            POINTS * NORMAL_POINTS, dtype=np.int64
        ),
        "original_path_internal_energy_only_calls_planned": np.asarray(
            N_IMAGES - 2, dtype=np.int64
        ),
        "total_energy_only_calls_planned": np.asarray(
            POINTS * NORMAL_POINTS + N_IMAGES - 2, dtype=np.int64
        ),
        "required_force_evaluations": np.asarray(0, dtype=np.int64),
        "optimization_N_F_contribution": np.asarray(0, dtype=np.int64),
        "selected_result": np.asarray(str(selected_result)),
        "selected_result_sha256": np.asarray(common.sha256_file(selected_result)),
    }
    atomic_npz_no_overwrite(output, payload)
    print(
        json.dumps(
            {
                "status": "PASS_ZERO_OPENMM_COORDINATE_FREEZE_R2",
                "output": str(output),
                "sha256": common.sha256_file(output),
                "surface_shape": list(coordinates.shape),
                "source_path_shape": list(source_path32.shape),
                "n_zero_exact": True,
                "physical_calls_started": 0,
                "surface_energy_only_calls_planned": POINTS * NORMAL_POINTS,
                "original_internal_energy_only_calls_planned": N_IMAGES - 2,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
