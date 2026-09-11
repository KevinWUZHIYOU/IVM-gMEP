#!/usr/bin/env python3
"""Audited LEPS Original String and OBSM (FIRE/L-BFGS) baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from leps_core import (
    EvaluationBudgetReached,
    LEPSBackend,
    TANGENT_DEFINITION,
    array_sha256,
    atomic_save_npz,
    audit_counter_consistency,
    counter_arrays,
    densified_barrier,
    evaluate_path,
    improved_tangents,
    load_common_path,
    make_evaluators,
    path_diagnostics,
    project_normal,
    project_relative,
    reference_saddle,
    write_json,
)


def cubic_arclength_reparameterize(path: np.ndarray) -> np.ndarray:
    """Interpolate the string at uniform cumulative-chord labels."""
    path = np.asarray(path, dtype=float)
    chord = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(chord)))
    if cumulative[-1] <= 0.0 or np.any(chord <= 0.0):
        raise ValueError("String must have positive adjacent chord lengths")
    labels = cumulative / cumulative[-1]
    result = CubicSpline(labels, path, axis=0)(np.linspace(0.0, 1.0, len(path)))
    result[0] = path[0]
    result[-1] = path[-1]
    return result


def lbfgs_direction(
    gradient: np.ndarray,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    initial_scale: float = 1.0,
) -> np.ndarray:
    """Return a normal-space L-BFGS descent direction."""
    q = np.asarray(gradient, dtype=float).copy()
    saved = []
    for s, y in reversed(pairs):
        rho = 1.0 / float(np.dot(s, y))
        alpha = rho * float(np.dot(s, q))
        saved.append((rho, alpha, y))
        q -= alpha * y
    if pairs:
        s, y = pairs[-1]
        scale = float(np.dot(s, y) / max(np.dot(y, y), 1e-30))
    else:
        scale = float(initial_scale)
    r = scale * q
    for (s, _), (rho, alpha, y) in zip(pairs, reversed(saved)):
        r += s * (alpha - rho * float(np.dot(y, r)))
    direction = -r
    if float(np.dot(direction, gradient)) >= 0.0:
        direction = -gradient
    return direction


def minimize_on_frozen_normal(
    reference,
    tangent,
    evaluator,
    *,
    solver,
    inner_steps,
    fire_dt,
    fire_beta,
    fire_mass,
    lbfgs_history,
    lbfgs_initial_scale,
    armijo_c1,
    backtrack,
    line_search_maxiter,
):
    position = np.asarray(reference, dtype=float).copy()
    energy, force = evaluator.query(position, "both")
    velocity = np.zeros_like(position)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    accepted = 0
    for _ in range(int(inner_steps)):
        gradient = project_normal(
            np.asarray([-force]), np.asarray([tangent])
        )[0]
        if np.linalg.norm(gradient) <= 1e-12:
            break
        if solver == "fire":
            norm_force = float(np.linalg.norm(force))
            acceleration = (
                force / float(fire_mass)
                - float(fire_beta) * velocity
                + float(fire_beta)
                * float(np.linalg.norm(velocity))
                * (force / norm_force if norm_force > 1e-30 else 0.0)
            )
            velocity += float(fire_dt) * acceleration
            displacement = project_normal(
                np.asarray([float(fire_dt) * velocity]),
                np.asarray([tangent]),
            )[0]
            trial = reference + project_normal(
                np.asarray([position + displacement - reference]),
                np.asarray([tangent]),
            )[0]
            if not np.all(np.isfinite(trial)):
                break
            try:
                trial_energy, trial_force = evaluator.query(trial, "both")
            except ValueError:
                break
            if (
                not np.isfinite(trial_energy)
                or not np.all(np.isfinite(trial_force))
            ):
                break
            energy, force = trial_energy, trial_force
            position = trial
            accepted += 1
            continue

        direction = project_normal(
            np.asarray(
                [lbfgs_direction(gradient, pairs, lbfgs_initial_scale)]
            ),
            np.asarray([tangent]),
        )[0]
        directional = float(np.dot(gradient, direction))
        if not np.all(np.isfinite(direction)) or not np.isfinite(directional):
            break
        step = 1.0
        accepted_trial = None
        for _ in range(int(line_search_maxiter)):
            trial = reference + project_normal(
                np.asarray([position + step * direction - reference]),
                np.asarray([tangent]),
            )[0]
            try:
                trial_energy, trial_force = evaluator.query(trial, "both")
            except ValueError:
                step *= float(backtrack)
                continue
            if (
                np.isfinite(trial_energy)
                and np.all(np.isfinite(trial_force))
                and trial_energy
                <= energy + float(armijo_c1) * step * directional
            ):
                accepted_trial = (trial, trial_energy, trial_force)
                break
            step *= float(backtrack)
        if accepted_trial is None:
            break
        trial, trial_energy, trial_force = accepted_trial
        new_gradient = -project_normal(
            np.asarray([trial_force]), np.asarray([tangent])
        )[0]
        s = trial - position
        y = new_gradient - gradient
        if float(np.dot(s, y)) > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
            pairs.append((s.copy(), y.copy()))
            pairs = pairs[-int(lbfgs_history) :]
        position, energy, force = trial, trial_energy, trial_force
        accepted += 1
    return position, accepted


def run(args):
    root = args.root.expanduser().resolve()
    initial_file = args.initial_path_file
    if not initial_file.is_absolute():
        initial_file = root / initial_file
    initial_path = load_common_path(initial_file)
    backend = LEPSBackend(label=args.run_label)
    evaluators = make_evaluators(backend, internal_limit=args.budget)
    current = initial_path.copy()
    saddle = reference_saddle()
    history = []
    stored_paths = []

    def record():
        energies, forces = evaluate_path(current, evaluators)
        diagnostic = path_diagnostics(current, energies, forces[1:-1])
        dense_barrier, _ = densified_barrier(current, subdivisions=80)
        counts = counter_arrays(evaluators)["joint_evaluations"]
        history.append(
            {
                "calls": int(np.max(counts[1:-1])),
                "i_perp": diagnostic["i_perp"],
                "i_arc": diagnostic["i_arc"],
                "raw_sparse_barrier": diagnostic["barrier"],
                "barrier": dense_barrier,
                "barrier_error": abs(
                    dense_barrier - saddle["forward_barrier"]
                ),
                "fmax": diagnostic["fmax"],
                "segment_length_cv": diagnostic["segment_length_cv"],
                "joint": counts.copy(),
            }
        )
        if args.save_all_paths or history[-1]["calls"] in {1, 2, 5, 10, 20, 50, 100, 200, 500}:
            stored_paths.append((history[-1]["calls"], current.copy()))
        return energies, forces

    stop_reason = "evaluation_budget"
    try:
        energies, forces = record()
        for outer in range(1, args.max_outer_iterations + 1):
            before_call = max(
                e.counters.joint_evaluations for e in evaluators[1:-1]
            )
            if before_call >= args.budget:
                break
            if args.method == "original":
                updated = current.copy()
                updated[1:-1] += float(args.dt) * forces[1:-1]
            else:
                tangents = improved_tangents(current, energies)
                minimizers = current.copy()
                for i in range(1, len(current) - 1):
                    minimizers[i], _ = minimize_on_frozen_normal(
                        current[i],
                        tangents[i - 1],
                        evaluators[i],
                        solver=args.solver,
                        inner_steps=args.inner_steps,
                        fire_dt=args.fire_dt,
                        fire_beta=args.fire_beta,
                        fire_mass=args.fire_mass,
                        lbfgs_history=args.lbfgs_history,
                        lbfgs_initial_scale=args.lbfgs_initial_scale,
                        armijo_c1=args.armijo_c1,
                        backtrack=args.backtrack,
                        line_search_maxiter=args.line_search_maxiter,
                    )
                updated = current.copy()
                updated[1:-1] = (
                    (1.0 - args.lambda_mix) * current[1:-1]
                    + args.lambda_mix * minimizers[1:-1]
                )
            if args.reparam_interval > 0 and outer % args.reparam_interval == 0:
                try:
                    updated = cubic_arclength_reparameterize(updated)
                except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                    stop_reason = "reparameterization_numerical_failure"
                    break
            current = updated
            energies, forces = record()
            after_call = max(
                e.counters.joint_evaluations for e in evaluators[1:-1]
            )
            if after_call == before_call:
                stop_reason = "stalled_no_new_evaluation_or_coordinate_change"
                break
    except EvaluationBudgetReached:
        stop_reason = "evaluation_budget"

    audit = audit_counter_consistency(backend, evaluators)
    if not audit["consistent"]:
        raise RuntimeError("String-baseline counter audit failed")
    configuration = {
        "method": args.method,
        "solver": args.solver if args.method == "obsm" else None,
        "run_label": args.run_label,
        "dt": args.dt if args.method == "original" else None,
        "lambda_mix": args.lambda_mix if args.method == "obsm" else None,
        "inner_steps": args.inner_steps if args.method == "obsm" else None,
        "fire_dt": args.fire_dt if args.solver == "fire" else None,
        "fire_beta": args.fire_beta if args.solver == "fire" else None,
        "fire_mass": args.fire_mass if args.solver == "fire" else None,
        "lbfgs_history": args.lbfgs_history if args.solver == "lbfgs" else None,
        "lbfgs_initial_scale": (
            args.lbfgs_initial_scale if args.solver == "lbfgs" else None
        ),
        "armijo_c1": args.armijo_c1 if args.solver == "lbfgs" else None,
        "backtrack": args.backtrack if args.solver == "lbfgs" else None,
        "reparam_interval": args.reparam_interval,
        "budget": args.budget,
        "initial_path_file": project_relative(initial_file, root),
        "initial_path_sha256": array_sha256(initial_path),
        "tangent_definition": TANGENT_DEFINITION,
        "barrier_definition": (
            "analytic LEPS energy on 80 linear subdivisions per image segment"
        ),
        "stop_reason": stop_reason,
    }
    config_sha = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()
    family = "original_string" if args.method == "original" else f"obsm_{args.solver}"
    output = root / "runs" / family / (
        f"{args.run_label}_cfg{config_sha[:12]}_budget{args.budget}.npz"
    )
    arrays = {
        key: np.asarray([row[key] for row in history])
        for key in (
            "calls",
            "i_perp",
            "i_arc",
            "barrier",
            "raw_sparse_barrier",
            "barrier_error",
            "fmax",
            "segment_length_cv",
        )
    }
    atomic_save_npz(
        output,
        method=np.asarray(family),
        solver=np.asarray(args.solver if args.method == "obsm" else "full_force_euler"),
        run_label=np.asarray(args.run_label),
        configuration_sha256=np.asarray(config_sha),
        initial_path=initial_path,
        final_path=current,
        initial_path_sha256=np.asarray(configuration["initial_path_sha256"]),
        calls_per_image=arrays.pop("calls"),
        joint_evaluations_history=np.asarray([row["joint"] for row in history]),
        stored_path_calls=np.asarray([item[0] for item in stored_paths], dtype=int),
        stored_paths=np.asarray([item[1] for item in stored_paths]),
        reference_barrier=np.asarray(saddle["forward_barrier"]),
        counter_audit_consistent=np.asarray(audit["consistent"]),
        **arrays,
    )
    write_json(
        output.with_suffix(".audit.json"),
        {
            "configuration": configuration,
            "configuration_sha256": config_sha,
            "counter_audit": audit,
            "result": project_relative(output, root),
        },
    )
    print(output)
    return output


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--initial-path-file",
        type=Path,
        default=Path("inputs/common_initial_path.npz"),
    )
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--method", choices=("original", "obsm"), required=True)
    parser.add_argument("--solver", choices=("fire", "lbfgs"), default="fire")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--lambda-mix", type=float, default=0.9)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--fire-dt", type=float, default=0.02)
    parser.add_argument("--fire-beta", type=float, default=0.1)
    parser.add_argument("--fire-mass", type=float, default=1.0)
    parser.add_argument("--lbfgs-history", type=int, default=7)
    parser.add_argument("--lbfgs-initial-scale", type=float, default=0.01)
    parser.add_argument("--armijo-c1", type=float, default=1e-4)
    parser.add_argument("--backtrack", type=float, default=0.5)
    parser.add_argument("--line-search-maxiter", type=int, default=12)
    parser.add_argument("--reparam-interval", type=int, default=1)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--max-outer-iterations", type=int, default=100000)
    parser.add_argument("--save-all-paths", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.dt <= 0 or args.fire_dt <= 0 or args.fire_mass <= 0:
        raise ValueError("Positive time step and FIRE mass required")
    if not 0.0 < args.lambda_mix < 1.0:
        raise ValueError("OBSM lambda_mix must lie in (0, 1)")
    if args.inner_steps <= 0 or args.lbfgs_history <= 0:
        raise ValueError("Positive inner_steps and lbfgs_history required")
    if args.reparam_interval < 0:
        raise ValueError("reparam_interval must be nonnegative")
    if not 0.0 < args.backtrack < 1.0:
        raise ValueError("backtrack must lie in (0, 1)")
    run(args)


if __name__ == "__main__":
    main()
