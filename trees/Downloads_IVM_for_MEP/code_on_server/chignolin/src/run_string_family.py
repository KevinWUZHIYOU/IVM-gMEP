#!/usr/bin/env python3
"""Audited Original String, OBSM-SD, and OBSM-CG runner.

The three methods in the comparison table of Samanta and E are implemented
without a displacement cap.  OBSM-SD and OBSM-CG support both inner stopping
interpretations expressly allowed by the paper: convergence of the projected
gradient and a fixed finite number of minimization steps.

``original``
    One explicit full physical-force steepest-descent step independently for
    every internal image, with no tangent projection, followed by the supplied
    Appendix cubic-spline reparameterization.  The main configuration uses
    every-step reparameterization; relaxed frequencies are explicit
    sensitivities.

``obsm_sd``
    Steepest-descent subminimization on each frozen normal hyperplane, OBSM
    mixing, and reparameterization every five string iterations.

``obsm_cg``
    Nonlinear-CG subminimization on every frozen normal hyperplane before
    mixing, followed by reparameterization every five string iterations.

The article and authors' public example page identify conjugate gradient and
GSL but do not identify Fletcher--Reeves versus Polak--Ribiere.  Both standard
variants are therefore available as an explicit sensitivity; neither is
silently described as uniquely specified by the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import line_search

from chignolin_core import (
    EvaluationBudgetReached,
    OpenMMBackend,
    TANGENT_DEFINITION,
    TOPOLOGY_FILENAME,
    array_sha256,
    atomic_save_npz,
    audit_counter_consistency,
    central_tangents,
    counter_arrays,
    evaluate_path,
    load_common_path,
    make_evaluators,
    path_diagnostics,
    write_json,
)
from run_obsm import (
    PAPER_ARXIV,
    PAPER_DOI,
    SPLINE_SCHEME,
    cubic_arclength_reparameterize,
    max_atomic_norm,
    project_one,
    projected_stop_norm,
)


PAPER_REPARAM_INTERVAL = 5
PAPER_STRING_LENGTH_TOL = 1e-5
AUTHOR_CODE_PAGE = (
    "https://web.math.princeton.edu/string/modified_implem.html"
)
METHOD_REVISION = (
    "paper_table_family_v3_exact_inner_source_snapshot_no_cap"
)
PATH_MILESTONES = {
    1,
    2,
    5,
    10,
    20,
    50,
    100,
    200,
    300,
    400,
    500,
}


def resolve_reparam_interval(
    method: str, requested_interval: int | None
) -> int:
    """Return the declared main frequency unless a sensitivity is requested."""
    if method not in {"original", "obsm_sd", "obsm_cg"}:
        raise ValueError(f"Unknown string-family method: {method}")
    if requested_interval is None:
        return 1 if method == "original" else PAPER_REPARAM_INTERVAL
    interval = int(requested_interval)
    if interval < 0:
        raise ValueError("reparam_interval must be nonnegative")
    return interval


def should_reparameterize(outer_iteration: int, interval: int) -> bool:
    """Use interval zero for the declared no-reparameterization sensitivity."""
    outer_iteration = int(outer_iteration)
    interval = int(interval)
    if outer_iteration <= 0:
        raise ValueError("outer_iteration must be positive")
    if interval < 0:
        raise ValueError("reparam_interval must be nonnegative")
    return interval > 0 and outer_iteration % interval == 0


def string_length(path: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def relative_string_length_change(old: np.ndarray, new: np.ndarray) -> float:
    new_length = string_length(new)
    if new_length <= 0.0:
        raise ValueError("String length must remain positive")
    return abs(string_length(new) - new_length) / new_length


def one_full_force_step(
    path: np.ndarray,
    internal_forces: np.ndarray,
    dt: float,
) -> np.ndarray:
    """One Original String step using each image's full physical force.

    Tangential motion is not projected out here.  The following string
    reparameterization redistributes the updated images along arclength.
    """
    path = np.asarray(path, dtype=float)
    internal_forces = np.asarray(internal_forces, dtype=float)
    if internal_forces.shape != path[1:-1].shape:
        raise ValueError("Internal force array has the wrong shape")
    updated = path.copy()
    updated[1:-1] += float(dt) * internal_forces
    return updated


@dataclass
class CGResult:
    position: np.ndarray
    energy: float
    force: np.ndarray
    iterations: int
    rejected_trials: int
    converged: bool
    stop_reason: str
    final_projected_fmax: float
    joint_evaluations_used: int


def subminimization_batch_complete(
    results: list[CGResult],
    *,
    expected_images: int,
    inner_mode: str,
    requested_steps: int,
) -> bool:
    """Require the declared stopping interpretation for every image."""
    if len(results) != int(expected_images):
        return False
    if inner_mode == "converge":
        return all(result.converged for result in results)
    if inner_mode == "few_steps":
        return all(
            result.iterations == int(requested_steps) for result in results
        )
    raise ValueError(f"Unknown inner_mode: {inner_mode}")


def minimize_sd_on_frozen_hyperplane(
    reference: np.ndarray,
    tangent: np.ndarray,
    evaluator,
    dt: float,
    inner_mode: str,
    inner_steps: int,
    inner_tolerance: float,
    inner_stop_norm: str,
    safety_max_iterations: int,
    reserve_evaluations: int = 1,
) -> CGResult:
    """Relax one frozen normal hyperplane by explicit steepest descent."""
    if inner_mode not in {"converge", "few_steps"}:
        raise ValueError("SD inner mode must be converge or few_steps")
    reference = np.asarray(reference, dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    before = int(evaluator.counters.joint_evaluations)
    energy, force = evaluator.query(reference, "both")
    position = reference.copy()
    force = np.asarray(force, dtype=float)
    gradient = project_one(-force, tangent)
    iterations = 0
    stop_reason = (
        "fixed_inner_steps"
        if inner_mode == "few_steps"
        else "safety_iteration_limit"
    )
    step_limit = (
        int(inner_steps)
        if inner_mode == "few_steps"
        else int(safety_max_iterations)
    )
    while iterations < step_limit:
        if (
            inner_mode == "converge"
            and projected_stop_norm(gradient, inner_stop_norm)
            <= float(inner_tolerance)
        ):
            stop_reason = "projected_gradient_tolerance"
            break
        if evaluator.max_joint_evaluations is not None:
            remaining = (
                int(evaluator.max_joint_evaluations)
                - int(evaluator.counters.joint_evaluations)
            )
            if remaining <= int(reserve_evaluations):
                stop_reason = "reserved_outer_evaluation"
                break
        displacement = -float(dt) * gradient
        if max_atomic_norm(displacement) <= 1e-30:
            stop_reason = "zero_direction"
            break
        trial = reference + project_one(
            position + displacement - reference, tangent
        )
        try:
            energy, force = evaluator.query(trial, "both")
        except EvaluationBudgetReached:
            stop_reason = "reserved_outer_evaluation"
            break
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            stop_reason = (
                f"numerical_failure_{type(exc).__name__}: {exc}"
            )
            break
        position = trial
        force = np.asarray(force, dtype=float)
        gradient = project_one(-force, tangent)
        iterations += 1
    converged = (
        projected_stop_norm(gradient, inner_stop_norm)
        <= float(inner_tolerance)
    )
    if inner_mode == "converge" and converged:
        stop_reason = "projected_gradient_tolerance"
    return CGResult(
        position=position,
        energy=float(energy),
        force=force,
        iterations=int(iterations),
        rejected_trials=0,
        converged=bool(converged),
        stop_reason=stop_reason,
        final_projected_fmax=max_atomic_norm(gradient),
        joint_evaluations_used=(
            int(evaluator.counters.joint_evaluations) - before
        ),
    )


def minimize_cg_on_frozen_hyperplane(
    reference: np.ndarray,
    tangent: np.ndarray,
    evaluator,
    variant: str,
    inner_fmax: float,
    wolfe_c1: float,
    wolfe_c2: float,
    line_search_maxiter: int,
    inner_mode: str,
    inner_steps: int,
    inner_stop_norm: str,
    safety_max_iterations: int,
    reserve_evaluations: int = 1,
) -> CGResult:
    """Relax one frozen OBSM hyperplane by standard nonlinear CG."""
    if variant not in {"fr", "pr_plus"}:
        raise ValueError("CG variant must be fr or pr_plus")
    if inner_mode not in {"converge", "few_steps"}:
        raise ValueError("CG inner mode must be converge or few_steps")
    reference = np.asarray(reference, dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    before = int(evaluator.counters.joint_evaluations)
    energy, force = evaluator.query(reference, "both")
    position = reference.copy()
    force = np.asarray(force, dtype=float)
    gradient = project_one(-force, tangent)
    direction = -gradient
    iterations = 0
    rejected = 0
    stop_reason = (
        "fixed_inner_steps"
        if inner_mode == "few_steps"
        else "safety_iteration_limit"
    )
    step_limit = (
        int(inner_steps)
        if inner_mode == "few_steps"
        else int(safety_max_iterations)
    )

    def constrained(coordinates: np.ndarray) -> np.ndarray:
        return reference + project_one(
            np.asarray(coordinates, dtype=float) - reference, tangent
        )

    def objective(coordinates: np.ndarray) -> float:
        return float(evaluator.query(constrained(coordinates), "both")[0])

    def objective_gradient(coordinates: np.ndarray) -> np.ndarray:
        raw_gradient = -np.asarray(
            evaluator.query(constrained(coordinates), "both")[1], dtype=float
        )
        return project_one(raw_gradient, tangent)

    while iterations < step_limit:
        if (
            inner_mode == "converge"
            and projected_stop_norm(gradient, inner_stop_norm)
            <= float(inner_fmax)
        ):
            stop_reason = "projected_gradient_tolerance"
            break
        if evaluator.max_joint_evaluations is not None:
            remaining = (
                int(evaluator.max_joint_evaluations)
                - int(evaluator.counters.joint_evaluations)
            )
            if remaining <= int(reserve_evaluations):
                stop_reason = "reserved_outer_evaluation"
                break
        if (
            not np.all(np.isfinite(direction))
            or float(np.dot(gradient, direction)) >= 0.0
        ):
            # Standard nonlinear-CG restart, not a displacement cap.
            direction = -gradient
        if max_atomic_norm(direction) <= 1e-30:
            stop_reason = "zero_direction"
            break
        before_line = int(evaluator.counters.joint_evaluations)
        try:
            alpha, _, _, _, _, _ = line_search(
                objective,
                objective_gradient,
                position,
                direction,
                gfk=gradient,
                old_fval=float(energy),
                c1=float(wolfe_c1),
                c2=float(wolfe_c2),
                maxiter=int(line_search_maxiter),
            )
        except (EvaluationBudgetReached, FloatingPointError, ValueError):
            alpha = None
        line_evaluations = (
            int(evaluator.counters.joint_evaluations) - before_line
        )
        if alpha is None:
            rejected += int(line_evaluations)
            stop_reason = "strong_wolfe_line_search_failed_or_budget_reserved"
            break
        new_position = constrained(position + float(alpha) * direction)
        try:
            new_energy, new_force = evaluator.query(new_position, "both")
        except EvaluationBudgetReached:
            stop_reason = "reserved_outer_evaluation"
            break
        new_force = np.asarray(new_force, dtype=float)
        new_gradient = project_one(-new_force, tangent)
        denominator = float(np.dot(gradient, gradient))
        if denominator <= 1e-30:
            beta = 0.0
        elif variant == "fr":
            beta = float(np.dot(new_gradient, new_gradient)) / denominator
        else:
            beta = max(
                0.0,
                float(np.dot(new_gradient, new_gradient - gradient))
                / denominator,
            )
        direction = project_one(-new_gradient + beta * direction, tangent)
        position = new_position
        energy = float(new_energy)
        force = new_force
        gradient = new_gradient
        iterations += 1
    converged = (
        projected_stop_norm(gradient, inner_stop_norm)
        <= float(inner_fmax)
    )
    if inner_mode == "converge" and converged:
        stop_reason = "projected_gradient_tolerance"
    return CGResult(
        position=position,
        energy=float(energy),
        force=np.asarray(force, dtype=float),
        iterations=int(iterations),
        rejected_trials=int(rejected),
        converged=bool(converged),
        stop_reason=stop_reason,
        final_projected_fmax=max_atomic_norm(gradient),
        joint_evaluations_used=(
            int(evaluator.counters.joint_evaluations) - before
        ),
    )


def configuration_payload(args) -> dict:
    method = str(args.method)
    return {
        "implementation_revision": METHOD_REVISION,
        "orchestration_source_snapshot_sha256": os.environ.get(
            "PAPER_FAMILY_SOURCE_SNAPSHOT_SHA256"
        ),
        "method": method,
        "dt": (
            float(args.dt) if method in {"original", "obsm_sd"} else None
        ),
        "original_string_descent_force": (
            "full physical force with no tangent projection"
            if method == "original"
            else None
        ),
        "lambda_mix": (
            None if method == "original" else float(args.lambda_mix)
        ),
        "inner_mode": (
            args.inner_mode
            if method in {"obsm_sd", "obsm_cg"}
            else None
        ),
        "fixed_inner_step_count_parameter": (
            int(args.inner_steps)
            if method in {"obsm_sd", "obsm_cg"}
            and args.inner_mode == "few_steps"
            else None
        ),
        "inner_stop_norm": (
            args.inner_stop_norm
            if method in {"obsm_sd", "obsm_cg"}
            else None
        ),
        "inner_projected_gradient_tolerance": (
            float(args.inner_fmax)
            if method in {"obsm_sd", "obsm_cg"}
            and args.inner_mode == "converge"
            else None
        ),
        "inner_safety_max_iterations": (
            int(args.inner_safety_max_iterations)
            if method in {"obsm_sd", "obsm_cg"}
            and args.inner_mode == "converge"
            else None
        ),
        "inner_safety_limit_is_tuned_parameter": False,
        "maximum_displacement_cap": None,
        "reparam_interval": int(args.reparam_interval),
        "paper_table_reparam_interval": PAPER_REPARAM_INTERVAL,
        "string_length_relative_tolerance": float(args.string_length_tol),
        "string_length_early_stop_enabled": bool(
            float(args.string_length_tol) > 0.0
        ),
        "cg_variant": args.cg_variant if method == "obsm_cg" else None,
        "cg_variant_uniquely_specified_by_paper": False,
        "cg_complete_relaxation": (
            method == "obsm_cg" and args.inner_mode == "converge"
        ),
        "cg_inner_fmax_ev_per_angstrom": (
            float(args.inner_fmax) if method == "obsm_cg" else None
        ),
        "line_search": (
            "SciPy strong Wolfe; implementation completion because the "
            "paper and public page do not identify the GSL CG subtype or "
            "line-search constants"
            if method == "obsm_cg"
            else None
        ),
        "wolfe_c1": float(args.wolfe_c1) if method == "obsm_cg" else None,
        "wolfe_c2": float(args.wolfe_c2) if method == "obsm_cg" else None,
        "line_search_maxiter": (
            int(args.line_search_maxiter) if method == "obsm_cg" else None
        ),
        "tangent": TANGENT_DEFINITION,
        "spline": SPLINE_SCHEME,
        "author_code_page": AUTHOR_CODE_PAGE,
    }


def run(args) -> Path:
    root = Path(args.root).expanduser().resolve()
    output_namespace = str(args.output_namespace)
    if (
        Path(output_namespace).name != output_namespace
        or output_namespace in {"", ".", ".."}
    ):
        raise ValueError("--output-namespace must be one directory-name component")
    initial_file = Path(args.initial_path_file).expanduser()
    if not initial_file.is_absolute():
        initial_file = root / initial_file
    initial_path = load_common_path(initial_file)
    backend = OpenMMBackend(
        root / "inputs" / TOPOLOGY_FILENAME,
        platform_preference=("CPU",),
        precision="double",
        label=f"{args.method}_optimization",
    )
    evaluators = make_evaluators(backend, internal_limit=int(args.budget))
    current = initial_path.copy()
    start_time = time.time()
    stored_paths = []
    stored_path_calls = []
    history = {
        name: []
        for name in (
            "calls_per_image",
            "mean_calls_per_internal_image",
            "total_internal_joint_evaluations",
            "physical_fmax",
            "i_perp",
            "barrier_kcal",
            "equidistance_integral",
            "i_arc",
            "max_unit_spring_force",
            "segment_length_cv",
            "segment_length_ratio",
            "wall_seconds",
            "outer_iteration",
            "relative_string_length_change",
            "reparameterized",
            "inner_iterations_sum",
            "inner_rejected_trials_sum",
            "inner_converged_images",
        )
    }
    history.update(
        {
            "counter_vectors": [],
            "energy_request_vectors": [],
            "force_request_vectors": [],
            "cache_hit_vectors": [],
            "inner_stop_reason_json": [],
        }
    )
    last = {
        "relative_length": np.nan,
        "reparameterized": False,
        "inner_iterations": 0,
        "inner_rejected": 0,
        "inner_converged": 0,
        "stop_reasons": {},
    }

    def record(outer_iteration: int) -> tuple[np.ndarray, np.ndarray]:
        energies, forces = evaluate_path(current, evaluators)
        diagnostic = path_diagnostics(current, energies, forces[1:-1])
        counters = counter_arrays(evaluators)
        internal = counters["joint_evaluations"][1:-1]
        values = {
            "calls_per_image": int(np.max(internal)),
            "mean_calls_per_internal_image": float(np.mean(internal)),
            "total_internal_joint_evaluations": int(np.sum(internal)),
            "physical_fmax": float(diagnostic["fmax"]),
            "i_perp": float(diagnostic["i_perp"]),
            "barrier_kcal": float(diagnostic["barrier_kcal"]),
            "equidistance_integral": float(
                diagnostic["equidistance_integral"]
            ),
            "i_arc": float(diagnostic["i_arc"]),
            "max_unit_spring_force": float(
                diagnostic["max_unit_spring_force"]
            ),
            "segment_length_cv": float(diagnostic["segment_length_cv"]),
            "segment_length_ratio": float(
                diagnostic["segment_length_ratio"]
            ),
            "wall_seconds": float(time.time() - start_time),
            "outer_iteration": int(outer_iteration),
            "relative_string_length_change": float(last["relative_length"]),
            "reparameterized": bool(last["reparameterized"]),
            "inner_iterations_sum": int(last["inner_iterations"]),
            "inner_rejected_trials_sum": int(last["inner_rejected"]),
            "inner_converged_images": int(last["inner_converged"]),
        }
        for name, value in values.items():
            history[name].append(value)
        history["counter_vectors"].append(
            counters["joint_evaluations"].copy()
        )
        history["energy_request_vectors"].append(
            counters["energy_requests"].copy()
        )
        history["force_request_vectors"].append(
            counters["force_requests"].copy()
        )
        history["cache_hit_vectors"].append(counters["cache_hits"].copy())
        call = int(values["calls_per_image"])
        if (
            args.save_all_paths
            or (args.save_all_until > 0 and call <= args.save_all_until)
            or call in PATH_MILESTONES
            or (
                args.path_interval > 0
                and call >= 500
                and call % args.path_interval == 0
            )
        ):
            stored_paths.append(current.copy())
            stored_path_calls.append(call)
        history["inner_stop_reason_json"].append(
            json.dumps(last["stop_reasons"], sort_keys=True)
        )
        print(
            f"[{args.method} dt={args.dt:g} lambda="
            f"{'none' if args.method == 'original' else f'{args.lambda_mix:g}'} "
            f"r={args.reparam_interval}] outer={outer_iteration:4d} "
            f"calls={values['calls_per_image']:4d} "
            f"Iperp={values['i_perp']:.6e} "
            f"barrier={values['barrier_kcal']:.4f} "
            f"fmax={values['physical_fmax']:.5f} "
            f"Iarc={values['i_arc']:.3e}",
            flush=True,
        )
        return energies, forces

    stop_reason = "evaluation_budget"
    outer = 0
    energies, forces = record(outer)
    cycle_phase_modulus = max(1, int(args.reparam_interval))
    initial_path_hash = array_sha256(np.asarray(current, dtype=float))
    seen_path_states = {(0, initial_path_hash): 0}
    path_hashes = [initial_path_hash]
    while True:
        internal_counts = counter_arrays(evaluators)["joint_evaluations"][1:-1]
        if int(np.max(internal_counts)) >= int(args.budget):
            break
        outer += 1
        reference = current.copy()
        last = {
            "relative_length": np.nan,
            "reparameterized": False,
            "inner_iterations": 0,
            "inner_rejected": 0,
            "inner_converged": 0,
            "stop_reasons": {},
        }
        if args.method == "original":
            proposed = one_full_force_step(
                reference,
                forces[1:-1],
                dt=float(args.dt),
            )
            last["inner_iterations"] = reference.shape[0] - 2
            last["inner_converged"] = 0
        else:
            tangents = central_tangents(reference)
            minimizers = reference.copy()
            results: list[CGResult] = []
            for local_index, image_index in enumerate(
                range(1, reference.shape[0] - 1)
            ):
                try:
                    if args.method == "obsm_sd":
                        result = minimize_sd_on_frozen_hyperplane(
                            reference=reference[image_index],
                            tangent=tangents[local_index],
                            evaluator=evaluators[image_index],
                            dt=args.dt,
                            inner_mode=args.inner_mode,
                            inner_steps=args.inner_steps,
                            inner_tolerance=args.inner_fmax,
                            inner_stop_norm=args.inner_stop_norm,
                            safety_max_iterations=(
                                args.inner_safety_max_iterations
                            ),
                            reserve_evaluations=1,
                        )
                    else:
                        result = minimize_cg_on_frozen_hyperplane(
                            reference=reference[image_index],
                            tangent=tangents[local_index],
                            evaluator=evaluators[image_index],
                            variant=args.cg_variant,
                            inner_fmax=args.inner_fmax,
                            wolfe_c1=args.wolfe_c1,
                            wolfe_c2=args.wolfe_c2,
                            line_search_maxiter=args.line_search_maxiter,
                            inner_mode=args.inner_mode,
                            inner_steps=args.inner_steps,
                            inner_stop_norm=args.inner_stop_norm,
                            safety_max_iterations=(
                                args.inner_safety_max_iterations
                            ),
                            reserve_evaluations=1,
                        )
                except EvaluationBudgetReached:
                    stop_reason = "evaluation_budget"
                    break
                minimizers[image_index] = result.position
                results.append(result)
            last["inner_iterations"] = int(
                sum(result.iterations for result in results)
            )
            last["inner_rejected"] = int(
                sum(result.rejected_trials for result in results)
            )
            last["inner_converged"] = int(
                sum(result.converged for result in results)
            )
            last["stop_reasons"] = {
                reason: int(sum(result.stop_reason == reason for result in results))
                for reason in sorted({result.stop_reason for result in results})
            }
            numerical_failure = any(
                result.stop_reason.startswith("numerical_failure_")
                for result in results
            )
            complete_batch = subminimization_batch_complete(
                results,
                expected_images=reference.shape[0] - 2,
                inner_mode=args.inner_mode,
                requested_steps=args.inner_steps,
            )
            if (
                not complete_batch
                or numerical_failure
            ):
                if numerical_failure:
                    stop_reason = f"{args.method}_numerical_failure"
                elif args.inner_mode == "converge":
                    stop_reason = (
                        f"{args.method}_subminimization_not_fully_converged"
                    )
                else:
                    stop_reason = (
                        f"{args.method}_fixed_inner_steps_not_completed"
                    )
                # All attempted/rejected calls remain visible on the cost axis.
                energies, forces = record(outer)
                break
            proposed = reference.copy()
            proposed[1:-1] = (
                (1.0 - float(args.lambda_mix)) * reference[1:-1]
                + float(args.lambda_mix) * minimizers[1:-1]
            )
        do_reparam = should_reparameterize(
            outer, int(args.reparam_interval)
        )
        try:
            current = (
                cubic_arclength_reparameterize(proposed)
                if do_reparam
                else proposed
            )
            current[0] = initial_path[0]
            current[-1] = initial_path[-1]
            last["reparameterized"] = bool(do_reparam)
            last["relative_length"] = relative_string_length_change(
                reference, current
            )
            energies, forces = record(outer)
        except EvaluationBudgetReached:
            current = reference
            stop_reason = "evaluation_budget_before_updated_path_evaluation"
            last["relative_length"] = np.nan
            last["reparameterized"] = False
            # The reference coordinates were already evaluated at the start
            # of every inner solve.  Recording them is therefore cache-only
            # and exposes all unsuccessful line-search/inner calls on the
            # optimization cost axis instead of silently dropping them.
            energies, forces = record(outer)
            break
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            current = reference
            stop_reason = (
                f"numerical_failure_{type(exc).__name__}: {exc}"
            )
            last["relative_length"] = np.nan
            last["reparameterized"] = False
            energies, forces = record(outer)
            break
        current_path_hash = array_sha256(np.asarray(current, dtype=float))
        state_key = (
            int(outer) % cycle_phase_modulus,
            current_path_hash,
        )
        if state_key in seen_path_states:
            # These solvers are stateless between outer iterations.  The
            # exact same path at the exact same reparameterization phase
            # therefore implies a deterministic future cycle.  This is an
            # exact repeat test, not a tuned displacement threshold.
            previous_outer = int(seen_path_states[state_key])
            cycle_hashes = path_hashes[previous_outer:] + [
                current_path_hash
            ]
            stop_reason = (
                "configured_inner_fixed_point"
                if len(set(cycle_hashes)) == 1
                else "deterministic_path_cycle"
            )
            break
        seen_path_states[state_key] = int(outer)
        path_hashes.append(current_path_hash)
        if (
            float(args.string_length_tol) > 0.0
            and float(last["relative_length"]) < float(args.string_length_tol)
        ):
            stop_reason = "relative_string_length_change"
            break
        if outer >= int(args.max_outer_iterations):
            stop_reason = "outer_iteration_safety_limit"
            break

    config = configuration_payload(args)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:12]
    lambda_tag = (
        "nolambda"
        if args.method == "original"
        else f"lam{args.lambda_mix:g}"
    )
    variant_tag = (
        f"_{args.cg_variant}" if args.method == "obsm_cg" else ""
    )
    tag = (
        f"{output_namespace}_{args.run_label}_{args.method}{variant_tag}_"
        f"{lambda_tag}_dt{args.dt:g}_r{args.reparam_interval}_"
        f"cfg{config_hash}_budget{args.budget}"
    ).replace(".", "p")
    output = root / "runs" / output_namespace / f"{tag}.npz"
    arrays = {
        name: np.asarray(values)
        for name, values in history.items()
    }
    final_call = int(history["calls_per_image"][-1])
    if not stored_path_calls or stored_path_calls[-1] != final_call:
        stored_paths.append(current.copy())
        stored_path_calls.append(final_call)
    atomic_save_npz(
        output,
        method=np.array(args.method),
        paper_doi=np.array(PAPER_DOI),
        paper_arxiv=np.array(PAPER_ARXIV),
        author_code_page=np.array(AUTHOR_CODE_PAGE),
        tangent_definition=np.array(TANGENT_DEFINITION),
        spline_definition=np.array(SPLINE_SCHEME),
        initial_path=initial_path,
        initial_path_file=np.array(str(initial_file)),
        initial_path_sha256=np.array(array_sha256(initial_path)),
        final_path=current,
        run_label=np.array(args.run_label),
        stop_reason=np.array(stop_reason),
        budget=np.array(args.budget),
        config_json=np.array(json.dumps(config, sort_keys=True)),
        config_hash=np.array(config_hash),
        stored_paths=np.asarray(stored_paths),
        stored_path_calls=np.asarray(stored_path_calls, dtype=int),
        path_interval=np.array(args.path_interval),
        save_all_until=np.array(args.save_all_until),
        **arrays,
    )
    audit = audit_counter_consistency(backend, evaluators)
    audit.update(
        {
            "result_file": str(output),
            "algorithm": args.method,
            "paper_doi": PAPER_DOI,
            "paper_arxiv": PAPER_ARXIV,
            "author_code_page": AUTHOR_CODE_PAGE,
            "configuration": config,
            "initial_path_sha256": array_sha256(initial_path),
            "contains_maximum_displacement_cap": False,
            "contains_fixed_n_inner_parameter": bool(
                args.method in {"obsm_sd", "obsm_cg"}
                and args.inner_mode == "few_steps"
            ),
            "contains_ivm_proximal_term": False,
            "reparameterization_openmm_calls": 0,
            "stop_reason": stop_reason,
            "final_call": int(history["calls_per_image"][-1]),
        }
    )
    write_json(output.with_suffix(".audit.json"), audit)
    print(output, flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--initial-path-file",
        default="inputs/common_initial_path_equidistant.npz",
    )
    parser.add_argument(
        "--method",
        choices=("original", "obsm_sd", "obsm_cg"),
        required=True,
    )
    parser.add_argument("--run-label", default="tuning")
    parser.add_argument(
        "--output-namespace", default="obsm_paper_table_family"
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--lambda-mix", type=float, default=0.25)
    parser.add_argument(
        "--reparam-interval",
        type=int,
        default=None,
        help=(
            "Default is 1 for Original String and the paper-table value 5 "
            "for OBSM; zero is a no-reparameterization sensitivity."
        ),
    )
    parser.add_argument(
        "--string-length-tol", type=float, default=PAPER_STRING_LENGTH_TOL
    )
    parser.add_argument("--cg-variant", choices=("fr", "pr_plus"), default="fr")
    parser.add_argument(
        "--inner-mode",
        choices=("converge", "few_steps"),
        default="few_steps",
    )
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--inner-fmax", type=float, default=1e-3)
    parser.add_argument(
        "--inner-stop-norm",
        choices=("l2", "max_atom"),
        default="l2",
    )
    parser.add_argument("--wolfe-c1", type=float, default=1e-4)
    parser.add_argument("--wolfe-c2", type=float, default=0.4)
    parser.add_argument("--line-search-maxiter", type=int, default=20)
    parser.add_argument(
        "--inner-safety-max-iterations", type=int, default=10000
    )
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--max-outer-iterations", type=int, default=100000)
    parser.add_argument("--save-all-paths", action="store_true")
    parser.add_argument("--save-all-until", type=int, default=0)
    parser.add_argument("--path-interval", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.reparam_interval = resolve_reparam_interval(
        args.method, args.reparam_interval
    )
    if args.dt <= 0.0:
        raise ValueError("dt must be positive")
    if not 0.0 < args.lambda_mix < 1.0:
        raise ValueError("OBSM mixing lambda must lie in the open interval (0,1)")
    if args.string_length_tol < 0.0:
        raise ValueError(
            "String-length convergence tolerance must be nonnegative; "
            "zero disables this paper-table early stop for fixed-budget curves"
        )
    if args.inner_fmax <= 0.0:
        raise ValueError("Projected-gradient tolerance must be positive")
    if args.inner_steps <= 0:
        raise ValueError("inner_steps must be positive")
    if not 0.0 < args.wolfe_c1 < args.wolfe_c2 < 1.0:
        raise ValueError("Strong-Wolfe constants require 0<c1<c2<1")
    if args.inner_safety_max_iterations <= 0:
        raise ValueError("Inner safety iteration limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
