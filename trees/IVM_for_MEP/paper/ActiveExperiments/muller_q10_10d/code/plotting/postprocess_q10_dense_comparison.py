#!/usr/bin/env python3
"""Apply one 5001-point improved-tangent certificate to all Muller-22 paths."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from q10_ivm_core import REFERENCE_BARRIER, improved_tangents_arbitrary
from q10_potential import Q, energy_gradient, energy_values


NN_PRODUCTION_REPORT = "reports/q10_spline_joint_production_locked.json"
DENSE_COUNT = 5001


def dense_metrics(path: np.ndarray, dense_count: int = DENSE_COUNT) -> tuple[float, float, float]:
    path = np.asarray(path, dtype=np.float64)
    nodes = np.linspace(0.0, 1.0, path.shape[0])
    sample = np.linspace(0.0, 1.0, int(dense_count))
    spline = CubicSpline(nodes, path, axis=0)
    points = spline(sample)
    energies = energy_values(points)
    tangent = improved_tangents_arbitrary(points, energies)
    gradients = np.asarray([energy_gradient(point)[1] for point in points[1:-1]])
    normal = gradients - tangent * np.sum(gradients * tangent, axis=1, keepdims=True)
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc_weights = 0.5 * (segments[:-1] + segments[1:])
    i_perp = float(np.sum(np.sum(normal * normal, axis=1) * arc_weights))
    delta_e = float(np.max(energies) - energies[0])
    inactive = float(np.max(np.abs(points @ Q.T)[:, 2:]))
    return i_perp, abs(delta_e - REFERENCE_BARRIER), inactive


def process_discrete(stage: Path, label: str, source: str) -> dict:
    archive = np.load(stage / source, allow_pickle=False)
    paths = np.asarray(archive["path_history"], dtype=np.float64)
    cost = np.asarray(archive["total_force_evaluations"], dtype=np.int64)
    metrics = np.asarray([dense_metrics(path) for path in paths])
    return save(stage, label, source, cost, metrics)


def process_neural(stage: Path, label: str, source: str) -> dict:
    archive = np.load(stage / source, allow_pickle=False)
    paths = np.asarray(archive["dense_sample_path"], dtype=np.float64)
    cost = np.asarray(archive["accepted_number_of_forces"], dtype=np.int64)
    metrics = np.asarray([dense_metrics(path) for path in paths])
    return save(stage, label, source, cost, metrics)


def process_locked_neural(stage: Path, label: str, source: str) -> dict:
    production = json.loads((stage / source).read_text(encoding="ascii"))
    replicate = production["results"][0]
    costs = []
    paths = []
    for index, stage_result in enumerate(replicate["stages"]):
        artifact = Path(stage_result["output"])
        artifact = stage.parent / artifact if artifact.parts[0] == stage.name else stage / artifact
        archive = np.load(artifact, allow_pickle=False)
        stage_cost = np.asarray(archive["accepted_number_of_forces"], dtype=np.int64)
        stage_paths = np.asarray(archive["dense_sample_path"], dtype=np.float64)
        start = 0 if index == 0 else 1
        costs.append(stage_cost[start:])
        paths.append(stage_paths[start:])
    cost = np.concatenate(costs)
    metrics = np.asarray([dense_metrics(path) for path in np.concatenate(paths, axis=0)])
    return save(stage, label, source, cost, metrics)


def save(stage: Path, label: str, source: str, cost: np.ndarray, metrics: np.ndarray) -> dict:
    output_dir = stage / "runs" / "dense_postprocessed_improved_5001"
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = label.lower().replace("-", "_").replace(" ", "_")
    output = output_dir / f"{slug}.npz"
    np.savez_compressed(
        output,
        method=np.asarray(label),
        source=np.asarray(source),
        number_of_forces=np.asarray(cost),
        dense_i_perp=metrics[:, 0],
        dense_delta_e_error=metrics[:, 1],
        dense_maximum_inactive_coordinate=metrics[:, 2],
        dense_count=np.asarray(DENSE_COUNT),
        interpolation=np.asarray("coordinate-wise not-a-knot cubic spline"),
        tangent=np.asarray("NEB Henkelman-Jonsson improved tangent"),
        delta_e_definition=np.asarray("max_s U(phi(s)) - U(phi(0))"),
        dense_information_enters_optimization=np.asarray(False),
    )
    within = np.asarray(cost) <= 5000
    return {
        "method": label,
        "source": source,
        "output": str(output.relative_to(stage)),
        "final_cost_within_5000": int(cost[within][-1]),
        "final_dense_i_perp_within_5000": float(metrics[within, 0][-1]),
        "minimum_dense_i_perp_within_5000": float(np.min(metrics[within, 0])),
        "final_dense_delta_e_error_within_5000": float(metrics[within, 1][-1]),
        "minimum_dense_delta_e_error_within_5000": float(np.min(metrics[within, 1])),
        "maximum_inactive_coordinate": float(np.max(metrics[:, 2])),
    }


def summarize_existing(stage: Path, label: str) -> dict:
    slug = label.lower().replace("-", "_").replace(" ", "_")
    output = stage / "runs" / "dense_postprocessed_improved_5001" / f"{slug}.npz"
    archive = np.load(output, allow_pickle=False)
    cost = np.asarray(archive["number_of_forces"], dtype=np.int64)
    dense_i = np.asarray(archive["dense_i_perp"], dtype=float)
    dense_barrier = np.asarray(archive["dense_delta_e_error"], dtype=float)
    inactive = np.asarray(archive["dense_maximum_inactive_coordinate"], dtype=float)
    within = cost <= 5000
    return {
        "method": label,
        "source": str(archive["source"].item()),
        "output": str(output.relative_to(stage)),
        "final_cost_within_5000": int(cost[within][-1]),
        "final_dense_i_perp_within_5000": float(dense_i[within][-1]),
        "minimum_dense_i_perp_within_5000": float(np.min(dense_i[within])),
        "final_dense_delta_e_error_within_5000": float(dense_barrier[within][-1]),
        "minimum_dense_delta_e_error_within_5000": float(np.min(dense_barrier[within])),
        "maximum_inactive_coordinate": float(np.max(inactive)),
    }


def main() -> None:
    stage = Path(__file__).resolve().parents[1]
    with (stage / "reports" / "q10_inexact_tuned_resample_summary.json").open() as handle:
        rows = json.load(handle)
    ivm = next(
        row for row in rows
        if np.isclose(row["config"]["alpha"], 1.0e-3)
        and row["stop_reason"] == "stationary"
    )
    with (stage / "reports" / "baseline_final_selection.json").open() as handle:
        selected = json.load(handle)["selected"]
    sources = [
        ("NN-IVM", NN_PRODUCTION_REPORT, True),
        ("IVM", ivm["output"], False),
        ("OBSM-FIRE", selected["obsm_fire"]["run"], False),
        ("OBSM-LBFGS", selected["obsm_lbfgs"]["run"], False),
        ("Original String", selected["original_string"]["run"], False),
        ("ODE-NEB", selected["ode_neb"]["run"], False),
        ("Static NEB", selected["static_neb"]["run"], False),
    ]
    summaries = [process_locked_neural(stage, *sources[0][:2])]
    summaries.extend(summarize_existing(stage, label) for label, _, _ in sources[1:])
    report = stage / "reports" / "q10_nn_ivm_dense_comparison.json"
    report.write_text(json.dumps({
        "case": "fixed-Q unperturbed 10D Muller-Brown, 22 images / 20 internal samples",
        "cost_axis": "direct sum of all uncached internal-image force calls",
        "dense_count": DENSE_COUNT,
        "interpolation": "coordinate-wise not-a-knot cubic spline",
        "tangent": "NEB Henkelman-Jonsson improved tangent",
        "delta_e_definition": "max_s U(phi(s)) - U(phi(0))",
        "reference_barrier": REFERENCE_BARRIER,
        "nn_selection_status": "production; theory and strict dense-metric gates passed",
        "nn_configuration": {
            "alpha": 1.0e-3,
            "feature_mode": "cubic B-spline",
            "locked_schedule": [
                {"force_interval": [0, 3200], "network_width": 12, "tangent_stencil_step": 0.002},
                {"force_interval": [3200, 3700], "network_width": 14, "tangent_stencil_step": 0.001},
                {"force_interval": [3700, 4000], "network_width": 8, "tangent_stencil_step": 0.0001},
            ],
            "force_samples_per_batch": 20,
            "inner_lbfgs_iterations": 3,
            "sampling_jitter": 0.3,
            "sampling_mode": "deterministic golden shift",
            "reproduction_seeds": [20260812, 20260813, 20260814],
            "unconstrained_delta": True,
            "lagrangian": False,
            "barrier_penalty": False,
            "dense_certificate_feedback": False,
        },
        "nn_cost_audit": {
            "number_of_forces": 4000,
            "definition": "direct sum of every one of the 20 sample-force streams",
            "energy_only_improved_tangent_queries": 2400,
            "energy_only_queries_are_not_counted_as_forces": True,
        },
        "runs": summaries,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    print(report)


if __name__ == "__main__":
    main()
