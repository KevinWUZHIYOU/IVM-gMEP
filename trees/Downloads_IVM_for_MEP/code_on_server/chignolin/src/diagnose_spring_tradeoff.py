#!/usr/bin/env python3
"""Measure the real OpenMM response to geometry-only spring redistribution.

This is a diagnostic, not an optimizer.  It starts from one stored path,
constructs central-difference spring-flow candidates without requesting
energy or force, and then evaluates every candidate with a separate audited
OpenMM evaluator set.  Dense barrier calls use another diagnostic evaluator
and are never mixed into the optimization cost axis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from chignolin_core import (
    ImageEvaluator,
    OpenMMBackend,
    central_tangents,
    densified_barrier,
    equidistance_diagnostics,
    evaluate_path,
    path_diagnostics,
    project_normal,
    spring_reparameterize,
)


def load_stored_path(result: Path, call: int) -> np.ndarray:
    with np.load(result, allow_pickle=False) as saved:
        calls = np.asarray(saved["stored_path_calls"], dtype=int)
        paths = np.asarray(saved["stored_paths"], dtype=float)
    matches = np.flatnonzero(calls == int(call))
    if matches.size == 0:
        raise ValueError(f"Call {call} is not stored in {result}")
    return paths[int(matches[-1])].copy()


def displacement_components(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    displacement = np.asarray(candidate[1:-1] - reference[1:-1], dtype=float)
    tangent = central_tangents(reference)
    normal = project_normal(displacement, tangent)
    tangential = displacement - normal
    n_atoms = reference.shape[1] // 3

    def global_rms(array: np.ndarray) -> float:
        return float(np.linalg.norm(array) / np.sqrt(array.shape[0] * n_atoms))

    total_norm = float(np.linalg.norm(displacement))
    return {
        "displacement_rms_a_per_atom": global_rms(displacement),
        "normal_displacement_rms_a_per_atom": global_rms(normal),
        "tangential_displacement_rms_a_per_atom": global_rms(tangential),
        "normal_displacement_fraction": (
            float(np.linalg.norm(normal) / total_norm)
            if total_norm > 0.0
            else 0.0
        ),
        "max_image_displacement_rms_a_per_atom": float(
            np.max(
                np.linalg.norm(displacement, axis=1)
                / np.sqrt(float(n_atoms))
            )
        ),
        "max_image_normal_displacement_rms_a_per_atom": float(
            np.max(np.linalg.norm(normal, axis=1) / np.sqrt(float(n_atoms)))
        ),
    }


def spring_euler_step(path: np.ndarray, dtau: float) -> np.ndarray:
    candidate = np.asarray(path, dtype=float).copy()
    segment_lengths = np.linalg.norm(np.diff(candidate, axis=0), axis=1)
    spring_force = (
        segment_lengths[1:] - segment_lengths[:-1]
    )[:, None] * central_tangents(candidate)
    candidate[1:-1] += float(dtau) * spring_force
    return candidate


def hermite_arclength_reparameterize(
    path: np.ndarray, dense_points: int = 20001
) -> np.ndarray:
    """Reach equal-chord spring equilibrium on a central-tangent curve."""
    from scipy.interpolate import CubicHermiteSpline
    from scipy.optimize import least_squares

    path = np.asarray(path, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    parameter = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    derivatives = np.empty_like(path)
    derivatives[1:-1] = central_tangents(path)
    derivatives[0] = (path[1] - path[0]) / segment_lengths[0]
    derivatives[-1] = (path[-1] - path[-2]) / segment_lengths[-1]
    spline = CubicHermiteSpline(parameter, path, derivatives, axis=0)
    span = float(parameter[-1] - parameter[0])
    initial_increments = np.diff(parameter)
    initial_logits = np.log(
        initial_increments[:-1] / initial_increments[-1]
    )

    def positions(logits):
        logits = np.concatenate((np.asarray(logits, dtype=float), [0.0]))
        logits -= np.max(logits)
        increments = np.exp(logits)
        increments *= span / np.sum(increments)
        node_parameter = parameter[0] + np.concatenate(
            ([0.0], np.cumsum(increments))
        )
        return np.asarray(spline(node_parameter), dtype=float)

    def residual(logits):
        candidate_lengths = np.linalg.norm(
            np.diff(positions(logits), axis=0), axis=1
        )
        return candidate_lengths[1:] - candidate_lengths[:-1]

    solution = least_squares(
        residual,
        initial_logits,
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=max(200, int(dense_points) // 10),
    )
    result = positions(solution.x)
    result[0] = path[0]
    result[-1] = path[-1]
    return result


def make_candidates(reference: np.ndarray):
    candidates: list[tuple[str, float, np.ndarray]] = [
        ("reference", 0.0, reference.copy())
    ]
    for dtau in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2):
        candidates.append(
            ("single_euler", float(dtau), spring_euler_step(reference, dtau))
        )

    relaxed, residual = spring_reparameterize(
        reference,
        spring_constant=1.0,
        dtau=0.05,
        max_iter=20000,
        tolerance=1e-10,
    )
    print(f"Full spring relaxation residual: {residual:.6e}", flush=True)
    direction = relaxed - reference
    for scale in (
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.35,
        0.50,
        0.75,
        1.00,
    ):
        candidate = reference.copy()
        candidate[1:-1] += float(scale) * direction[1:-1]
        candidates.append(("relaxed_direction", float(scale), candidate))

    hermite = hermite_arclength_reparameterize(reference)
    hermite_direction = hermite - reference
    for scale in (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00):
        candidate = reference.copy()
        candidate[1:-1] += float(scale) * hermite_direction[1:-1]
        candidates.append(("hermite_arclength", float(scale), candidate))
    return candidates


def evaluate_candidates(
    root: Path,
    reference: np.ndarray,
    candidates,
    dense_subdivisions: int,
):
    topology = root / "inputs" / "chignolin_clean.pdb"
    backend = OpenMMBackend(
        topology,
        platform_preference=("CPU",),
        precision="double",
        label="spring_tradeoff_path_diagnostic",
    )
    evaluators = [
        ImageEvaluator(backend, image_index=i)
        for i in range(reference.shape[0])
    ]
    dense_backend = OpenMMBackend(
        topology,
        platform_preference=("CPU",),
        precision="double",
        label="spring_tradeoff_dense_diagnostic",
    )
    dense_evaluator = ImageEvaluator(dense_backend, image_index=0)

    rows = []
    for index, (family, parameter, candidate) in enumerate(candidates):
        before = backend.raw_joint_evaluations
        energies, forces = evaluate_path(candidate, evaluators)
        diagnostic = path_diagnostics(candidate, energies, forces[1:-1])
        path_calls = int(backend.raw_joint_evaluations - before)
        dense_before = dense_backend.raw_joint_evaluations
        dense_barrier, _ = densified_barrier(
            candidate, dense_evaluator, dense_subdivisions
        )
        dense_calls = int(dense_backend.raw_joint_evaluations - dense_before)
        row = {
            "index": int(index),
            "family": family,
            "parameter": float(parameter),
            "i_perp": float(diagnostic["i_perp"]),
            "physical_fmax": float(diagnostic["fmax"]),
            "sparse_barrier_kcal": float(diagnostic["barrier_kcal"]),
            "dense_barrier_kcal": float(dense_barrier),
            "equidistance_integral": float(
                diagnostic["equidistance_integral"]
            ),
            "equidistance_relative_defect": float(
                diagnostic["equidistance_relative_defect"]
            ),
            "segment_length_cv": float(diagnostic["segment_length_cv"]),
            "segment_length_ratio": float(
                diagnostic["segment_length_ratio"]
            ),
            "path_joint_evaluations": path_calls,
            "dense_joint_evaluations": dense_calls,
        }
        row.update(displacement_components(reference, candidate))
        rows.append(row)
        print(
            f"{index + 1:02d}/{len(candidates):02d} "
            f"{family}={parameter:g}: "
            f"I={row['i_perp']:.6g}, "
            f"eq={row['equidistance_integral']:.6g}, "
            f"CV={row['segment_length_cv']:.5f}, "
            f"barrier{dense_subdivisions}={dense_barrier:.6f}",
            flush=True,
        )

    path_cache_misses = sum(
        evaluator.counters.joint_evaluations for evaluator in evaluators
    )
    if backend.raw_joint_evaluations != path_cache_misses:
        raise RuntimeError(
            "Path diagnostic backend/cache audit mismatch: "
            f"{backend.raw_joint_evaluations} != {path_cache_misses}"
        )
    if (
        dense_backend.raw_joint_evaluations
        != dense_evaluator.counters.joint_evaluations
    ):
        raise RuntimeError(
            "Dense diagnostic backend/cache audit mismatch: "
            f"{dense_backend.raw_joint_evaluations} != "
            f"{dense_evaluator.counters.joint_evaluations}"
        )
    audit = {
        "path_backend_joint_evaluations": int(
            backend.raw_joint_evaluations
        ),
        "path_cache_misses": int(path_cache_misses),
        "path_energy_requests": int(
            sum(e.counters.energy_requests for e in evaluators)
        ),
        "path_force_requests": int(
            sum(e.counters.force_requests for e in evaluators)
        ),
        "path_cache_hits": int(
            sum(e.counters.cache_hits for e in evaluators)
        ),
        "dense_backend_joint_evaluations": int(
            dense_backend.raw_joint_evaluations
        ),
        "dense_cache_misses": int(
            dense_evaluator.counters.joint_evaluations
        ),
        "dense_energy_requests": int(
            dense_evaluator.counters.energy_requests
        ),
        "dense_force_requests": int(
            dense_evaluator.counters.force_requests
        ),
        "dense_cache_hits": int(dense_evaluator.counters.cache_hits),
        "diagnostic_counts_excluded_from_optimization_axis": True,
    }
    return rows, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--call", type=int, default=500)
    parser.add_argument("--output-label", required=True)
    parser.add_argument("--dense-subdivisions", type=int, default=10)
    parser.add_argument(
        "--families",
        default="",
        help="Optional comma-separated candidate-family filter.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    result = Path(args.result).expanduser().resolve()
    reference = load_stored_path(result, args.call)
    candidates = make_candidates(reference)
    if args.families:
        requested = {
            item.strip()
            for item in args.families.split(",")
            if item.strip()
        }
        candidates = [
            candidate
            for candidate in candidates
            if candidate[0] in requested
        ]
        if not candidates:
            raise ValueError(
                f"No candidates match requested families: {sorted(requested)}"
            )
    rows, audit = evaluate_candidates(
        root, reference, candidates, args.dense_subdivisions
    )

    output_dir = root / "runs" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_label}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / f"{args.output_label}.audit.json"
    payload = {
        "source_result": str(result),
        "source_call": int(args.call),
        "central_tangent_only": True,
        "dense_subdivisions": int(args.dense_subdivisions),
        "rows": rows,
        "audit": audit,
    }
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, json_path)
    print(f"Saved {csv_path}", flush=True)
    print(f"Saved {json_path}", flush=True)


if __name__ == "__main__":
    main()
