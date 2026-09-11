#!/usr/bin/env python3
"""Fail-closed validation for the completed LEPS study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from leps_core import TANGENT_DEFINITION, array_sha256


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def scalar(data: dict, key: str):
    return np.asarray(data[key]).item()


def validate_run(path: Path, initial_hash: str) -> None:
    data = load(path)
    assert str(scalar(data, "initial_path_sha256")) == initial_hash
    assert array_sha256(np.asarray(data["initial_path"])) == initial_hash
    assert str(scalar(data, "tangent_definition")) == TANGENT_DEFINITION
    assert bool(scalar(data, "counter_audit_consistent"))
    counter = np.asarray(data["counter_vector"][-1], dtype=int)
    backend = int(scalar(data, "backend_raw_joint_evaluations"))
    assert int(np.sum(counter)) == backend
    calls = np.asarray(data["calls_per_image"], dtype=int)
    assert np.all(np.diff(calls) >= 0)
    assert int(calls[-1]) == int(np.max(counter[1:-1]))
    for key in (
        "i_perp",
        "fmax",
        "dense_barrier",
        "barrier_error",
        "i_arc",
        "equidistance_integral",
        "spring_force_max_unit",
        "segment_length_cv",
    ):
        assert np.all(np.isfinite(data[key])), f"{path}: nonfinite {key}"
        assert np.all(np.asarray(data[key]) >= 0.0), f"{path}: negative {key}"


def validate_continuous_run(path: Path, initial_hash: str) -> None:
    data = load(path)
    assert str(scalar(data, "initial_path_sha256")) == initial_hash
    assert array_sha256(np.asarray(data["initial_path"])) == initial_hash
    assert str(scalar(data, "tangent_definition")) == (
        "normalized torch autograd dphi/ds"
    )
    assert bool(scalar(data, "counter_audit_consistent"))
    counter = np.asarray(data["counter_vector"][-1], dtype=int)
    backend = int(scalar(data, "backend_raw_joint_evaluations"))
    assert int(np.sum(counter)) == backend
    collocation_count = int(scalar(data, "collocation_count"))
    assert collocation_count == 20
    assert counter.size == collocation_count + 2
    calls = np.asarray(data["calls_per_image"], dtype=int)
    assert np.all(np.diff(calls) >= 0)
    assert int(calls[-1]) == int(np.max(counter[1:-1]))
    for key in (
        "dense_i_perp",
        "dense_fmax",
        "dense_barrier",
        "dense_barrier_error",
        "path_length",
        "minimum_speed",
        "spring_force_max_unit",
        "segment_length_cv",
    ):
        assert np.all(np.isfinite(data[key])), f"{path}: nonfinite {key}"
        assert np.all(np.asarray(data[key]) >= 0.0), f"{path}: negative {key}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    with np.load(root / "inputs" / "common_initial_path.npz") as saved:
        initial = np.asarray(saved["initial_path"])
    initial_hash = array_sha256(initial)

    neb_scan = sorted(
        (root / "runs" / "neb").glob("neb_scan_*_budget500.npz")
    )
    ivm_grid = sorted(
        (root / "runs" / "ivm").glob("ivm_alpha_*_budget*.npz")
    )
    assert len(neb_scan) == 11, f"expected 11 NEB scans, got {len(neb_scan)}"
    assert len(ivm_grid) == 60, f"expected 60 IVM grid runs, got {len(ivm_grid)}"
    near_exact = [
        load(path) for path in ivm_grid if "_near_exact_" in path.name
    ]
    assert len(near_exact) == 20
    kkt_fraction = np.concatenate(
        [
            np.asarray(data["subproblem_kkt_fraction"], dtype=float)[
                np.isfinite(data["subproblem_kkt_fraction"])
            ]
            for data in near_exact
        ]
    )
    assert float(np.mean(kkt_fraction)) >= 0.99

    summary = json.loads(
        (root / "reports" / "selection_summary.json").read_text()
    )
    assert summary["common_initial_path"]["strictly_common"]
    assert summary["tangent"]["strictly_common"]
    selected_paths = [
        root / summary["production_results"]["IVM"],
        root / summary["production_results"]["ODE-NEB"],
    ]
    formal = neb_scan + ivm_grid + selected_paths
    for path in formal:
        validate_run(path, initial_hash)

    continuous_paths = (
        root
        / "runs"
        / "continuous_v3"
        / "nn_ivm_adaptive_al_a100_j015_b500.npz",
        root
        / "runs"
        / "continuous_v2"
        / "gmam_autodiff20_production_lr001_b500.npz",
    )
    for path in continuous_paths:
        validate_continuous_run(path, initial_hash)
    continuous_summary = json.loads(
        (root / "reports" / "continuous_methods_summary.json").read_text()
    )
    assert continuous_summary["common_initial_path_hash_count"] == 1
    assert (
        continuous_summary["methods"]["NN–IVM"]["final_dense_i_perp"]
        < 0.11
    )
    assert (
        continuous_summary["methods"]["NN–IVM"]["minimum_barrier_error"]
        < 3e-4
    )
    gmam = load(continuous_paths[1])
    assert "no explicit or FD Hessian" in str(
        scalar(gmam, "hessian_definition")
    )

    expected_stems = (
        "leps_ivm_vs_neb",
        "leps_cost_to_precision",
        "leps_alpha_sensitivity",
        "leps_alpha_trajectories",
        "leps_subproblem_reparam_ablation",
        "leps_path_evolution_1x4",
        "leps_neb_k_scan",
        "leps_counter_audit",
        "leps_fmax",
        "leps_continuous_methods_comparison",
        "leps_continuous_per_stream_comparison",
        "leps_continuous_cost_to_precision",
        "leps_continuous_path_evolution_1x4",
        "leps_continuous_counter_audit",
    )
    for stem in expected_stems:
        for suffix in ("pdf", "svg", "png"):
            path = root / "figures" / f"{stem}.{suffix}"
            assert path.exists() and path.stat().st_size > 0

    print(
        json.dumps(
            {
                "validated_runs": len(formal) + len(continuous_paths),
                "initial_path_sha256": initial_hash,
                "selected_alpha": summary["selected_ivm"]["alpha"],
                "selected_neb_k": summary["selected_neb"]["k"],
                "status": "PASS",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
