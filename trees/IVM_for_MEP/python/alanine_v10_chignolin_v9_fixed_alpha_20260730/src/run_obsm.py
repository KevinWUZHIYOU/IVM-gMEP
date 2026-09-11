#!/usr/bin/env python3
"""Faithful, audited Optimization-Based String Method (OBSM) runner.

This implementation follows Samanta and E (2012):

1. freeze the normalized central-difference tangent of the current string;
2. minimize the *bare* potential energy on each normal hyperplane;
3. mix the old image and the constrained minimizer with ``lambda_mix``;
4. optionally redistribute the string by cubic-spline arclength interpolation.

The constrained subproblem contains no IVM proximal term and no global
``I_perp`` acceptance gate.  Every energy request obtains energy and force
together through the per-image run-local evaluator in :mod:`ala_core`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from scipy.optimize import line_search

from ala_core import (
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
from forcefield_preconditioner import ForceFieldPreconditionerBank


PAPER_DOI = "10.4208/cicp.220212.030812a"
PAPER_ARXIV = "1009.5612"
OBSM_OBJECTIVE = "bare OpenMM potential U constrained to frozen normal hyperplane"
SPLINE_SCHEME = (
    "Simplified String Method Appendix A cubic spline with sigma tridiagonal "
    "system and endpoint third derivatives from the first/last four-point "
    "interpolating cubics"
)
FIRE_SCHEME = (
    "explicit Euler discretization of OBSM Section-3 FIRE: full physical force "
    "updates velocity and only dx/dt is projected onto the frozen hyperplane"
)
BFGS_SCHEME = (
    "OBSM Section-3 full dense BFGS on one frozen normal hyperplane: initialize "
    "H0 and G0=H0^-1 once at x0, carry Hk and Gk through the complete inner-k "
    "iteration, update G by the Sherman-Morrison-derived inverse-BFGS recursion, "
    "obtain p=G F, use a strong-Wolfe line search along the declared paper-p or "
    "projected-p direction, set "
    "x_new=x+alpha*p_perp, s=alpha*p and y=-F_new+F_old, and independently "
    "apply the printed full-Hessian update for consistency auditing"
)
REPARAMETERIZATION = (
    f"cumulative Cartesian chord length sampled at uniform labels; {SPLINE_SCHEME}"
)
BUDGET_BOUNDARY_POLICY = (
    "stop before mixing/reparameterization when an outer cycle performs no "
    "joint evaluation, accepts/rejects no trial, and not all images converged"
)
BFGS_IMPLEMENTATION_REVISION = (
    "sm_inverse_v5_exact_inner_source_snapshot"
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


def project_one(vector: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Project one flattened Cartesian vector onto the tangent-normal plane."""
    vector = np.asarray(vector, dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    return vector - float(np.dot(vector, tangent)) * tangent


def max_atomic_norm(vector: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(np.asarray(vector).reshape(-1, 3), axis=1)))


def projected_stop_norm(vector: np.ndarray, norm_definition: str) -> float:
    vector = np.asarray(vector, dtype=float)
    if norm_definition == "l2":
        return float(np.linalg.norm(vector))
    if norm_definition == "max_atom":
        return max_atomic_norm(vector)
    raise ValueError("Projected stopping norm must be l2 or max_atom")


def third_divided_difference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the third divided difference of exactly four vector-valued data."""
    x = np.asarray(x, dtype=float)
    values = np.asarray(y, dtype=float).copy()
    if x.shape != (4,) or values.shape[0] != 4:
        raise ValueError("Third divided difference requires exactly four nodes")
    for order in range(1, 4):
        denominator = x[order:] - x[:-order]
        values = (values[1:] - values[:-1]) / denominator.reshape(
            (-1,) + (1,) * (values.ndim - 1)
        )
    return np.asarray(values[0], dtype=float)


def appendix_spline_sigmas(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve the Appendix-A symmetric tridiagonal system for ``sigma``.

    The spline convention is exactly

    ``s = w*y[i+1] + (1-w)*y[i] + h**2*((w**3-w)*sigma[i+1]
    + ((1-w)**3-(1-w))*sigma[i])``.

    Interior rows impose first-derivative continuity.  The two boundary rows
    prescribe the spline third derivative using the cubic interpolants through
    the first and last four nodes, respectively.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape[0] != x.size:
        raise ValueError("x must be one-dimensional and match y's first axis")
    if x.size < 4:
        raise ValueError("Appendix-A endpoint conditions require at least four nodes")
    h = np.diff(x)
    if np.any(h <= 0.0):
        raise ValueError("Spline nodes must be strictly increasing")

    count = x.size
    matrix = np.zeros((count, count), dtype=float)
    rhs = np.zeros_like(y, dtype=float)

    # Multiplying each endpoint third-derivative equation by h makes the
    # complete coefficient matrix symmetric and tridiagonal.
    left_dd3 = third_divided_difference(x[:4], y[:4])
    matrix[0, 0] = -h[0]
    matrix[0, 1] = h[0]
    rhs[0] = h[0] ** 2 * left_dd3

    slopes = np.diff(y, axis=0) / h.reshape(
        (-1,) + (1,) * (y.ndim - 1)
    )
    for i in range(1, count - 1):
        matrix[i, i - 1] = h[i - 1]
        matrix[i, i] = 2.0 * (h[i - 1] + h[i])
        matrix[i, i + 1] = h[i]
        rhs[i] = slopes[i] - slopes[i - 1]

    right_dd3 = third_divided_difference(x[-4:], y[-4:])
    matrix[-1, -2] = h[-1]
    matrix[-1, -1] = -h[-1]
    rhs[-1] = -(h[-1] ** 2) * right_dd3

    flattened_rhs = rhs.reshape(count, -1)
    sigma = np.linalg.solve(matrix, flattened_rhs)
    return sigma.reshape(y.shape)


def appendix_cubic_spline(
    x: np.ndarray, y: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Evaluate the exact cubic spline specified in Appendix A."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 1:
        raise ValueError("Spline targets must be one-dimensional")
    if np.any(targets < x[0]) or np.any(targets > x[-1]):
        raise ValueError("Spline targets must lie within the node interval")
    sigma = appendix_spline_sigmas(x, y)
    interval = np.searchsorted(x, targets, side="right") - 1
    interval = np.clip(interval, 0, x.size - 2)
    h = x[interval + 1] - x[interval]
    w = (targets - x[interval]) / h
    wbar = 1.0 - w
    broadcast_shape = (-1,) + (1,) * (y.ndim - 1)
    w_b = w.reshape(broadcast_shape)
    wbar_b = wbar.reshape(broadcast_shape)
    h2_b = (h**2).reshape(broadcast_shape)
    return (
        w_b * y[interval + 1]
        + wbar_b * y[interval]
        + h2_b
        * (
            (w_b**3 - w_b) * sigma[interval + 1]
            + (wbar_b**3 - wbar_b) * sigma[interval]
        )
    )


def cubic_arclength_reparameterize(path: np.ndarray) -> np.ndarray:
    """Redistribute by the exact Appendix-A spline at equal chord labels."""
    path = np.asarray(path, dtype=float)
    lengths = np.linalg.norm(path[1:] - path[:-1], axis=1)
    if np.any(lengths <= 1e-14):
        raise ValueError("Cubic reparameterization requires distinct neighboring images")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    cumulative /= cumulative[-1]
    result = appendix_cubic_spline(
        cumulative, path, np.linspace(0.0, 1.0, path.shape[0])
    )
    result[0] = path[0]
    result[-1] = path[-1]
    return np.asarray(result, dtype=float)


def full_bfgs_update(
    hessian: np.ndarray,
    paper_s: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Apply the Hessian update printed in OBSM Section 3.

    The article defines ``s_k = alpha_k p_k`` even though the coordinate
    update uses ``alpha_k p_k_perp``.  We intentionally preserve that exact
    distinction here rather than silently replacing ``s_k`` by the actual
    projected displacement.
    """
    hessian = np.asarray(hessian, dtype=float)
    paper_s = np.asarray(paper_s, dtype=float)
    y = np.asarray(y, dtype=float)
    ys = float(np.dot(y, paper_s))
    hs = hessian @ paper_s
    shs = float(np.dot(paper_s, hs))
    if abs(ys) <= 1e-30 or abs(shs) <= 1e-30:
        raise FloatingPointError("singular denominator in the paper BFGS update")
    updated = (
        hessian
        + np.outer(y, y) / ys
        - np.outer(hs, hs) / shs
    )
    return 0.5 * (updated + updated.T)


def sherman_morrison_inverse_bfgs_update(
    inverse_hessian: np.ndarray,
    paper_s: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Update ``H^-1`` by the Sherman--Morrison BFGS inverse recursion.

    The article asks that the inverse used in step (i) be obtained with the
    Sherman--Morrison scheme.  Applying that scheme to the rank-two Hessian
    update printed in step (v) gives

    ``G+ = (I-rho*s*y.T) G (I-rho*y*s.T) + rho*s*s.T``,

    where ``G=H^-1`` and ``rho=1/(y.T*s)``.  The paper's unprojected
    ``s=alpha*p`` is intentionally used here.
    """
    inverse_hessian = np.asarray(inverse_hessian, dtype=float)
    paper_s = np.asarray(paper_s, dtype=float)
    y = np.asarray(y, dtype=float)
    ys = float(np.dot(y, paper_s))
    if abs(ys) <= 1e-30:
        raise FloatingPointError(
            "singular denominator in Sherman-Morrison inverse-BFGS update"
        )
    rho = 1.0 / ys
    identity = np.eye(inverse_hessian.shape[0], dtype=float)
    left = identity - rho * np.outer(paper_s, y)
    right = identity - rho * np.outer(y, paper_s)
    updated = (
        left @ inverse_hessian @ right
        + rho * np.outer(paper_s, paper_s)
    )
    if not np.all(np.isfinite(updated)):
        raise FloatingPointError(
            "non-finite Sherman-Morrison inverse-BFGS update"
        )
    return 0.5 * (updated + updated.T)


def finite_difference_openmm_hessian(
    position: np.ndarray,
    evaluator,
    step_angstrom: float,
) -> np.ndarray:
    """Estimate ``nabla^2 V`` from counted central force differences.

    Every displaced-coordinate query is routed through the image-local joint
    energy/force evaluator, so all ``2 * ndof`` OpenMM evaluations are part of
    the optimization cost.  No regularization, positive-definite projection,
    pseudoinverse, or displacement cap is applied.
    """
    position = np.asarray(position, dtype=float)
    step = float(step_angstrom)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Hessian finite-difference step must be positive")
    ndof = int(position.size)
    hessian = np.empty((ndof, ndof), dtype=float)
    for coordinate in range(ndof):
        displacement = np.zeros(ndof, dtype=float)
        displacement[coordinate] = step
        force_plus = np.asarray(
            evaluator.query(position + displacement, "force"), dtype=float
        )
        force_minus = np.asarray(
            evaluator.query(position - displacement, "force"), dtype=float
        )
        # F=-grad(V), hence Hessian(V)=-dF/dx.
        hessian[:, coordinate] = -(
            force_plus - force_minus
        ) / (2.0 * step)
    if not np.all(np.isfinite(hessian)):
        raise FloatingPointError("non-finite finite-difference OpenMM Hessian")
    return 0.5 * (hessian + hessian.T)


@dataclass
class InnerResult:
    position: np.ndarray
    energy: float
    force: np.ndarray
    accepted_steps: int
    rejected_trials: int
    converged: bool
    stop_reason: str
    initial_energy: float
    final_projected_fmax: float
    joint_evaluations_used: int


@dataclass
class PersistentBFGSState:
    """Full BFGS curvature state for one frozen-hyperplane minimization.

    ``H0`` is initialized exactly once for this state.  The paper unambiguously
    carries the state through its inner ``k`` loop.  Whether it should also be
    reused after outer OBSM mixing/reparameterization changes the hyperplane is
    not stated; the runner therefore exposes that choice as an audited
    sensitivity rather than silently assuming one interpretation.
    """

    hessian: np.ndarray | None = None
    inverse_hessian: np.ndarray | None = None
    h0_initializations: int = 0
    h0_joint_evaluations: int = 0
    accepted_updates: int = 0
    failed_updates: int = 0
    h0_mode: str | None = None
    initialized_position: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        return self.hessian is not None and self.inverse_hessian is not None

    def initialize(
        self,
        position: np.ndarray,
        evaluator,
        h0_mode: str,
        hessian_fd_step: float,
        h0_matrix: np.ndarray | None = None,
    ) -> None:
        if self.initialized or self.h0_initializations != 0:
            raise RuntimeError("Persistent BFGS H0 may be initialized only once")
        before = int(evaluator.counters.joint_evaluations)
        if h0_mode == "finite_difference":
            hessian = finite_difference_openmm_hessian(
                position, evaluator, hessian_fd_step
            )
            inverse_hessian = np.linalg.inv(hessian)
        elif h0_mode == "identity":
            hessian = np.eye(np.asarray(position).size, dtype=float)
            inverse_hessian = np.eye(np.asarray(position).size, dtype=float)
        elif h0_mode == "bonded_forcefield":
            if h0_matrix is None:
                raise ValueError(
                    "bonded_forcefield H0 requires a supplied matrix"
                )
            hessian = np.asarray(h0_matrix, dtype=float)
            if hessian.shape != (
                np.asarray(position).size,
                np.asarray(position).size,
            ):
                raise ValueError("Bonded force-field H0 has wrong shape")
            inverse_hessian = np.linalg.inv(hessian)
        else:
            raise ValueError(f"Unknown BFGS H0 mode: {h0_mode}")
        self.hessian = np.asarray(hessian, dtype=float)
        self.inverse_hessian = np.asarray(inverse_hessian, dtype=float)
        self.h0_initializations = 1
        self.h0_joint_evaluations = (
            int(evaluator.counters.joint_evaluations) - before
        )
        self.h0_mode = str(h0_mode)
        self.initialized_position = np.asarray(position, dtype=float).copy()


def obsm_subminimization_batch_complete(
    results: list[InnerResult],
    *,
    expected_images: int,
    inner_mode: str,
    requested_steps: int,
) -> bool:
    """Require every image to satisfy the declared inner stopping branch."""
    if len(results) != int(expected_images):
        return False
    if inner_mode == "converge":
        return all(result.converged for result in results)
    if inner_mode == "few_steps":
        return all(
            result.accepted_steps == int(requested_steps)
            for result in results
        )
    raise ValueError(f"Unknown inner_mode: {inner_mode}")


def minimize_on_frozen_hyperplane(
    reference: np.ndarray,
    tangent: np.ndarray,
    evaluator,
    solver: str,
    max_steps: int,
    inner_fmax: float,
    wolfe_c1: float,
    wolfe_c2: float,
    line_search_maxiter: int,
    bfgs_h0_mode: str = "identity",
    hessian_fd_step: float = 1e-4,
    bfgs_h0_matrix: np.ndarray | None = None,
    bfgs_line_search_direction: str = "projected_p",
    bfgs_line_search_fallback: str = "stop",
    armijo_maxiter: int = 20,
    armijo_shrink: float = 0.5,
    fire_mass: float = 1.0,
    fire_beta: float = 0.1,
    fire_dt: float = 0.02,
    reserve_evaluations: int = 1,
    bfgs_state: PersistentBFGSState | None = None,
    inner_stop_norm: str = "l2",
    inner_mode: str = "converge",
) -> InnerResult:
    """Approximately minimize bare U on one frozen OBSM normal hyperplane."""
    if inner_mode not in {"converge", "few_steps"}:
        raise ValueError("inner_mode must be converge or few_steps")
    reference = np.asarray(reference, dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    start_count = int(evaluator.counters.joint_evaluations)
    energy, force = evaluator.query(reference, "both")
    initial_energy = float(energy)
    position = reference.copy()
    gradient = np.asarray(-force, dtype=float)
    projected_gradient = project_one(gradient, tangent)
    fire_velocity = np.zeros_like(reference)
    accepted = 0
    rejected = 0
    stop_reason = "inner_step_limit"
    if solver == "bfgs":
        if bfgs_state is None:
            raise ValueError(
                "BFGS requires a run-persistent per-image PersistentBFGSState"
            )
        if (
            bfgs_state.initialized
            and bfgs_state.h0_mode != str(bfgs_h0_mode)
        ):
            raise ValueError(
                "Cannot change BFGS H0 mode after persistent state initialization"
            )
        if not bfgs_state.initialized:
            try:
                bfgs_state.initialize(
                    reference,
                    evaluator,
                    bfgs_h0_mode,
                    hessian_fd_step,
                    bfgs_h0_matrix,
                )
            except EvaluationBudgetReached:
                stop_reason = "budget_during_finite_difference_h0"
            except np.linalg.LinAlgError:
                stop_reason = "singular_finite_difference_h0"
            except (FloatingPointError, ValueError) as exc:
                stop_reason = (
                    f"numerical_failure_{type(exc).__name__}: {exc}"
                )
            if stop_reason != "inner_step_limit":
                return InnerResult(
                    position=position,
                    energy=float(energy),
                    force=np.asarray(force, dtype=float),
                    accepted_steps=accepted,
                    rejected_trials=rejected,
                    converged=False,
                    stop_reason=stop_reason,
                    initial_energy=initial_energy,
                    final_projected_fmax=max_atomic_norm(projected_gradient),
                    joint_evaluations_used=(
                        int(evaluator.counters.joint_evaluations) - start_count
                    ),
                )
        hessian = bfgs_state.hessian
        inverse_hessian = bfgs_state.inverse_hessian
        if hessian is None or inverse_hessian is None:
            raise RuntimeError("Persistent BFGS state failed to initialize")
        if bfgs_line_search_direction not in ("paper_p", "projected_p"):
            raise ValueError(
                "BFGS line-search direction must be paper_p or projected_p"
            )

    for _ in range(int(max_steps)):
        if (
            inner_mode == "converge"
            and projected_stop_norm(projected_gradient, inner_stop_norm)
            <= float(inner_fmax)
        ):
            stop_reason = "projected_fmax"
            break
        if evaluator.max_joint_evaluations is not None:
            remaining = (
                int(evaluator.max_joint_evaluations)
                - int(evaluator.counters.joint_evaluations)
            )
            if remaining <= int(reserve_evaluations):
                stop_reason = "reserved_outer_evaluation"
                break

        if solver == "fire":
            # Samanta--E Section 3 evolves v with the *full* physical force,
            # while only dx/dt = v - (tau,v)tau is projected.  Projecting the
            # force or velocity here would define a different FIRE equation.
            physical_force = np.asarray(force, dtype=float)
            force_norm = float(np.linalg.norm(physical_force))
            force_hat = (
                physical_force / force_norm
                if force_norm > 1e-30
                else np.zeros_like(physical_force)
            )
            acceleration = (
                physical_force / float(fire_mass)
                - float(fire_beta) * fire_velocity
                + float(fire_beta)
                * float(np.linalg.norm(fire_velocity))
                * force_hat
            )
            fire_velocity = fire_velocity + float(fire_dt) * acceleration
            displacement = project_one(
                float(fire_dt) * fire_velocity, tangent
            )
            if max_atomic_norm(displacement) <= 1e-30:
                stop_reason = "zero_direction"
                break
            trial = reference + project_one(
                position + displacement - reference, tangent
            )
            try:
                trial_energy, trial_force = evaluator.query(trial, "both")
            except EvaluationBudgetReached:
                stop_reason = "reserved_outer_evaluation"
                break
            except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                stop_reason = (
                    f"numerical_failure_{type(exc).__name__}: {exc}"
                )
                break
            position = trial
            energy = float(trial_energy)
            force = np.asarray(trial_force, dtype=float)
            gradient = np.asarray(-force, dtype=float)
            projected_gradient = project_one(gradient, tangent)
            accepted += 1
            continue
        if solver != "bfgs":
            raise ValueError(f"Unknown OBSM inner solver: {solver}")

        # Paper step (i): obtain H^{-1} through Sherman--Morrison and then
        # p_k = H_k^{-1} F(x_k).  Production code deliberately does not
        # replace this with np.linalg.solve(H, F).
        paper_direction = inverse_hessian @ np.asarray(force, dtype=float)
        direction = project_one(paper_direction, tangent)
        search_direction = (
            paper_direction
            if bfgs_line_search_direction == "paper_p"
            else direction
        )
        directional_derivative = float(np.dot(gradient, search_direction))
        if (
            not np.all(np.isfinite(direction))
            or not np.all(np.isfinite(search_direction))
            or directional_derivative >= 0.0
        ):
            stop_reason = "non_descent_bfgs_direction"
            break
        if max_atomic_norm(direction) <= 1e-30:
            stop_reason = "zero_direction"
            break

        def constrained(coordinates: np.ndarray) -> np.ndarray:
            return reference + project_one(
                np.asarray(coordinates, dtype=float) - reference, tangent
            )

        def line_coordinates(coordinates: np.ndarray) -> np.ndarray:
            if bfgs_line_search_direction == "paper_p":
                return np.asarray(coordinates, dtype=float)
            return constrained(coordinates)

        def objective(coordinates: np.ndarray) -> float:
            return float(evaluator.query(line_coordinates(coordinates), "both")[0])

        def objective_gradient(coordinates: np.ndarray) -> np.ndarray:
            return -np.asarray(
                evaluator.query(line_coordinates(coordinates), "both")[1],
                dtype=float,
            )

        before_line_search = int(evaluator.counters.joint_evaluations)
        try:
            alpha, _, _, _, _, _ = line_search(
                objective,
                objective_gradient,
                position,
                search_direction,
                gfk=gradient,
                old_fval=float(energy),
                c1=float(wolfe_c1),
                c2=float(wolfe_c2),
                maxiter=int(line_search_maxiter),
            )
        except (EvaluationBudgetReached, FloatingPointError, ValueError):
            alpha = None
        line_evaluations = (
            int(evaluator.counters.joint_evaluations) - before_line_search
        )
        if alpha is None:
            rejected += int(line_evaluations)
            if bfgs_line_search_fallback == "armijo":
                trial_alpha = 1.0
                for _ in range(int(armijo_maxiter)):
                    if evaluator.max_joint_evaluations is not None:
                        remaining = (
                            int(evaluator.max_joint_evaluations)
                            - int(evaluator.counters.joint_evaluations)
                        )
                        if remaining <= int(reserve_evaluations):
                            break
                    try:
                        trial_energy = objective(
                            position + trial_alpha * search_direction
                        )
                    except (
                        EvaluationBudgetReached,
                        FloatingPointError,
                        ValueError,
                    ):
                        break
                    if trial_energy <= (
                        float(energy)
                        + float(wolfe_c1)
                        * trial_alpha
                        * directional_derivative
                    ):
                        alpha = trial_alpha
                        break
                    rejected += 1
                    trial_alpha *= float(armijo_shrink)
            if alpha is None:
                stop_reason = (
                    "strong_wolfe_and_fallback_line_search_failed_or_"
                    "budget_reserved"
                )
                break

        # Paper step (iii) uses the projected direction even when step (ii)
        # obtains alpha along the unprojected p_k.
        new_position = constrained(
            position + float(alpha) * paper_direction
        )
        try:
            new_energy, new_force = evaluator.query(new_position, "both")
        except EvaluationBudgetReached:
            stop_reason = "reserved_outer_evaluation"
            break
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            stop_reason = (
                f"numerical_failure_{type(exc).__name__}: {exc}"
            )
            break
        new_energy = float(new_energy)
        new_force = np.asarray(new_force, dtype=float)
        new_gradient = -new_force
        y = new_gradient - gradient
        # These are the printed full-BFGS definitions, including its
        # unprojected s_k.
        paper_s = float(alpha) * paper_direction
        try:
            next_hessian = full_bfgs_update(hessian, paper_s, y)
            next_inverse_hessian = sherman_morrison_inverse_bfgs_update(
                inverse_hessian, paper_s, y
            )
        except FloatingPointError:
            bfgs_state.failed_updates += 1
            stop_reason = "singular_paper_bfgs_update"
            position = new_position
            energy = new_energy
            force = new_force
            gradient = new_gradient
            projected_gradient = project_one(gradient, tangent)
            accepted += 1
            break
        hessian = next_hessian
        inverse_hessian = next_inverse_hessian
        bfgs_state.hessian = next_hessian
        bfgs_state.inverse_hessian = next_inverse_hessian
        bfgs_state.accepted_updates += 1
        position = new_position
        energy = new_energy
        force = new_force
        gradient = new_gradient
        projected_gradient = project_one(gradient, tangent)
        accepted += 1
    else:
        stop_reason = "inner_step_limit"

    converged = (
        projected_stop_norm(projected_gradient, inner_stop_norm)
        <= float(inner_fmax)
    )
    if inner_mode == "converge" and converged:
        stop_reason = "projected_fmax"
    used = int(evaluator.counters.joint_evaluations) - start_count
    return InnerResult(
        position=position,
        energy=float(energy),
        force=np.asarray(force, dtype=float),
        accepted_steps=accepted,
        rejected_trials=rejected,
        converged=converged,
        stop_reason=stop_reason,
        initial_energy=initial_energy,
        final_projected_fmax=max_atomic_norm(projected_gradient),
        joint_evaluations_used=used,
    )


def configuration_payload(args) -> dict:
    bfgs_selected = args.solver == "bfgs"
    inner_mode = getattr(args, "inner_mode", "few_steps")
    h0_mode = getattr(args, "bfgs_h0", "identity")
    line_direction = getattr(
        args, "bfgs_line_search_direction", "paper_p"
    )
    state_scope = getattr(args, "bfgs_state_scope", "paper_local")
    failure_policy = getattr(args, "bfgs_failure_policy", "stop")
    line_search_fallback = getattr(
        args, "bfgs_line_search_fallback", "stop"
    )
    disclosures = [
        "fixed standard strong-Wolfe constants because line-search details are omitted",
        "explicit Euler FIRE discretization because the ODE integrator is omitted",
        (
            "projected-force convergence tolerance is an implementation "
            "choice because the paper does not state a numerical tolerance"
        ),
    ]
    if h0_mode == "identity":
        disclosures.insert(
            0,
            (
                "identity BFGS H0 replaces the printed nabla^2 V because "
                "OpenMM does not expose a Hessian through this force API"
            ),
        )
    elif h0_mode == "finite_difference":
        disclosures.insert(
            0,
            (
                "central finite-difference step for the printed nabla^2 V "
                "is an implementation choice; every displaced force is counted"
            ),
        )
    else:
        disclosures.insert(
            0,
            (
                "bonded force-field H0 is a disclosed mechanism ablation: "
                "it gives OBSM the same call-free initial Hessian model as "
                "IVM and is not labeled as the strict paper baseline"
            ),
        )
    if bfgs_selected and failure_policy == "restart_curvature":
        disclosures.insert(
            0,
            (
                "after an incomplete fixed-step BFGS image batch, preserve the "
                "unmoved string and every spent force call, discard all "
                "cross-outer curvature states, and restart from the declared H0"
            ),
        )
    if bfgs_selected and line_search_fallback == "armijo":
        disclosures.insert(
            0,
            (
                "when SciPy strong Wolfe returns no step, use a counted "
                "Armijo backtracking line search along the same declared "
                "direction; the OBSM paper leaves line-search details unspecified"
            ),
        )
    payload = {
        "implementation_revision": BFGS_IMPLEMENTATION_REVISION,
        "orchestration_source_snapshot_sha256": os.environ.get(
            "PAPER_FAMILY_SOURCE_SNAPSHOT_SHA256"
        ),
        "solver": args.solver,
        "lambda_mix": float(args.lambda_mix),
        "inner_mode": inner_mode,
        "inner_steps": (
            int(args.inner_steps) if inner_mode == "few_steps" else None
        ),
        "inner_safety_max_steps": (
            int(getattr(args, "max_inner_steps", 10000))
            if inner_mode == "converge"
            else None
        ),
        "reparam_interval": int(args.reparam_interval),
        "maximum_displacement_cap": None,
        "inner_fmax": float(args.inner_fmax),
        "inner_stop_norm": getattr(args, "inner_stop_norm", "l2"),
        "inner_stopping_rule": (
            f"norm={getattr(args, 'inner_stop_norm', 'l2')} of the complete "
            "projected gradient/force vector"
        ),
        "bfgs_h0": (
            (
                "identity, initialized once per newly frozen-hyperplane solve"
                if state_scope == "paper_local"
                else (
                    "identity, initialized once per image and carried across "
                    "outer string updates"
                )
            )
            if h0_mode == "identity"
            else (
                (
                    "central finite-difference OpenMM nabla^2 V once at the "
                    "start of each newly frozen-hyperplane solve"
                    if state_scope == "paper_local"
                    else (
                        "central finite-difference OpenMM nabla^2 V once at "
                        "each image's global initial coordinate, then carried "
                        "across outer string updates"
                    )
                )
                + "; no regularization"
            )
            if h0_mode == "finite_difference"
            else (
                "bonded force-field Hessian model with c_stab=1; "
                "same call-free initial curvature model as IVM"
            )
        ),
        "bfgs_h0_initialization_frequency": (
            (
                "once at the start of every newly frozen-hyperplane BFGS "
                "minimization"
                if state_scope == "paper_local"
                else (
                    "once per internal image for the complete OBSM run; "
                    "cross-outer curvature warm start"
                )
            )
        ),
        "bfgs_state_scope": state_scope,
        "bfgs_failure_policy": failure_policy,
        "bfgs_max_consecutive_restarts": int(
            getattr(args, "bfgs_max_consecutive_restarts", 3)
        ),
        "bfgs_state_reset_after_subminimization": state_scope == "paper_local",
        "bfgs_state_reset_after_mixing": state_scope == "paper_local",
        "bfgs_state_reset_after_reparameterization": state_scope == "paper_local",
        "bfgs_state_reset_after_incomplete_fixed_step_batch": bool(
            bfgs_selected
            and state_scope == "curvature_carry"
            and failure_policy == "restart_curvature"
        ),
        "bfgs_cross_outer_state_is_paper_specified": False,
        "bfgs_h0_mode": h0_mode,
        "bfgs_hessian_fd_step_angstrom": (
            float(getattr(args, "hessian_fd_step", 1e-4))
            if h0_mode == "finite_difference"
            else None
        ),
        "bfgs_bonded_forcefield_h0_call_free": (
            h0_mode == "bonded_forcefield"
        ),
        "bfgs_bonded_h0_cache": (
            str(getattr(args, "bfgs_bonded_h0_cache", "")) or None
        ),
        "bfgs_bonded_h0_cache_semantics": (
            "precomputed from the identical run initial path; coordinate-only "
            "construction with zero OpenMM energy/force evaluations"
            if getattr(args, "bfgs_bonded_h0_cache", "")
            else None
        ),
        "bfgs_hessian_storage": "full dense",
        "bfgs_inverse_storage": "full dense",
        "bfgs_direction": "p = G @ F with G = H^-1",
        "bfgs_inverse_update": (
            "Sherman-Morrison-derived inverse-BFGS recursion; no direct "
            "linear solve in production"
        ),
        "bfgs_printed_hessian_update_retained": True,
        "bfgs_sherman_morrison_direction": bool(bfgs_selected),
        "bfgs_h0_matches_printed_nabla2V": (
            h0_mode == "finite_difference"
            if bfgs_selected
            else None
        ),
        "bfgs_line_search_direction": line_direction,
        "bfgs_scheme": BFGS_SCHEME,
        "line_search": f"SciPy strong Wolfe along {line_direction}",
        "wolfe_c1": float(args.wolfe_c1),
        "wolfe_c2": float(args.wolfe_c2),
        "line_search_maxiter": int(args.line_search_maxiter),
        "bfgs_line_search_fallback": line_search_fallback,
        "armijo_maxiter": int(getattr(args, "armijo_maxiter", 20)),
        "armijo_shrink": float(getattr(args, "armijo_shrink", 0.5)),
        "fire_mass": float(args.fire_mass),
        "fire_beta": float(args.fire_beta),
        "fire_dt": float(args.fire_dt),
        "fire_scheme": FIRE_SCHEME,
        "spline_scheme": SPLINE_SCHEME,
        "strict_original_equations": not bfgs_selected,
        "paper_steps_i_to_v_except_h0_and_unspecified_line_search": bool(
            bfgs_selected
        ),
        "implementation_only_disclosures": disclosures,
    }
    return payload


def run(args) -> Path:
    root = Path(args.root).expanduser().resolve()
    output_namespace = str(args.output_namespace)
    if (
        Path(output_namespace).name != output_namespace
        or output_namespace in ("", ".", "..")
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
        label="obsm_optimization",
    )
    evaluators = make_evaluators(backend, internal_limit=args.budget)
    ffprecon_bank = (
        ForceFieldPreconditionerBank(
            backend,
            root / "inputs" / TOPOLOGY_FILENAME,
            c_stab=1.0,
        )
        if args.solver == "bfgs"
        and args.bfgs_h0 == "bonded_forcefield"
        else None
    )
    bonded_h0_cache_path = None
    if getattr(args, "bfgs_bonded_h0_cache", ""):
        if args.bfgs_h0 != "bonded_forcefield":
            raise ValueError("--bfgs-bonded-h0-cache requires bonded_forcefield H0")
        if args.bfgs_state_scope != "curvature_carry":
            raise ValueError(
                "A fixed initial-path H0 cache is valid only with curvature_carry"
            )
        bonded_h0_cache_path = Path(args.bfgs_bonded_h0_cache).expanduser()
        if not bonded_h0_cache_path.is_absolute():
            bonded_h0_cache_path = root / bonded_h0_cache_path
        with np.load(bonded_h0_cache_path, allow_pickle=False) as cached:
            if str(cached["initial_path_sha256"].item()) != array_sha256(initial_path):
                raise RuntimeError("Bonded H0 cache initial-path hash mismatch")
            ffprecon_bank.load_dense_hessians(cached["dense_hessians"])
    persistent_bfgs_states = (
        [PersistentBFGSState() for _ in range(initial_path.shape[0] - 2)]
        if args.solver == "bfgs" and args.bfgs_state_scope == "curvature_carry"
        else []
    )
    bfgs_state_archive: list[list[PersistentBFGSState]] = [
        [] for _ in range(initial_path.shape[0] - 2)
    ]
    if persistent_bfgs_states:
        for local_index, state in enumerate(persistent_bfgs_states):
            bfgs_state_archive[local_index].append(state)
    current = initial_path.copy()
    bfgs_restart_count = 0
    consecutive_bfgs_restarts = 0
    start_time = time.time()
    stored_paths = []
    stored_path_calls = []

    scalar_names = (
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
        "inner_accepted_steps_sum",
        "inner_rejected_trials_sum",
        "inner_converged_images",
        "inner_bare_energy_nonincrease_images",
        "reparameterized",
        "bfgs_h0_initializations_sum",
        "bfgs_h0_joint_evaluations_sum",
        "bfgs_accepted_updates_sum",
        "bfgs_failed_updates_sum",
        "bfgs_curvature_restarts",
    )
    history = {name: [] for name in scalar_names}
    history.update(
        {
            "counter_vectors": [],
            "energy_request_vectors": [],
            "force_request_vectors": [],
            "cache_hit_vectors": [],
            "inner_stop_reason_json": [],
        }
    )
    last_inner_summary = {
        "accepted": 0,
        "rejected": 0,
        "converged": 0,
        "energy_nonincrease": 20,
        "reparameterized": False,
        "stop_reasons": {},
    }

    def all_bfgs_states() -> list[PersistentBFGSState]:
        return [
            state
            for image_states in bfgs_state_archive
            for state in image_states
        ]

    def record(outer_iteration: int):
        energies, forces = evaluate_path(current, evaluators)
        diagnostic = path_diagnostics(current, energies, forces[1:-1])
        counters = counter_arrays(evaluators)
        internal = counters["joint_evaluations"][1:-1]
        history["calls_per_image"].append(int(np.max(internal)))
        history["mean_calls_per_internal_image"].append(float(np.mean(internal)))
        history["total_internal_joint_evaluations"].append(int(np.sum(internal)))
        history["physical_fmax"].append(float(diagnostic["fmax"]))
        history["i_perp"].append(float(diagnostic["i_perp"]))
        history["barrier_kcal"].append(float(diagnostic["barrier_kcal"]))
        history["equidistance_integral"].append(
            float(diagnostic["equidistance_integral"])
        )
        history["i_arc"].append(float(diagnostic["i_arc"]))
        history["max_unit_spring_force"].append(
            float(diagnostic["max_unit_spring_force"])
        )
        history["segment_length_cv"].append(float(diagnostic["segment_length_cv"]))
        history["segment_length_ratio"].append(
            float(diagnostic["segment_length_ratio"])
        )
        history["wall_seconds"].append(float(time.time() - start_time))
        history["outer_iteration"].append(int(outer_iteration))
        history["inner_accepted_steps_sum"].append(last_inner_summary["accepted"])
        history["inner_rejected_trials_sum"].append(last_inner_summary["rejected"])
        history["inner_converged_images"].append(last_inner_summary["converged"])
        history["inner_bare_energy_nonincrease_images"].append(
            last_inner_summary["energy_nonincrease"]
        )
        history["reparameterized"].append(last_inner_summary["reparameterized"])
        history["bfgs_h0_initializations_sum"].append(
            int(sum(state.h0_initializations for state in all_bfgs_states()))
        )
        history["bfgs_h0_joint_evaluations_sum"].append(
            int(sum(state.h0_joint_evaluations for state in all_bfgs_states()))
        )
        history["bfgs_accepted_updates_sum"].append(
            int(sum(state.accepted_updates for state in all_bfgs_states()))
        )
        history["bfgs_failed_updates_sum"].append(
            int(sum(state.failed_updates for state in all_bfgs_states()))
        )
        history["bfgs_curvature_restarts"].append(int(bfgs_restart_count))
        history["counter_vectors"].append(counters["joint_evaluations"].copy())
        history["energy_request_vectors"].append(
            counters["energy_requests"].copy()
        )
        history["force_request_vectors"].append(counters["force_requests"].copy())
        history["cache_hit_vectors"].append(counters["cache_hits"].copy())
        call = int(np.max(internal))
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
            json.dumps(last_inner_summary["stop_reasons"], sort_keys=True)
        )
        inner_label = (
            str(args.inner_steps)
            if args.inner_mode == "few_steps"
            else f"converge@{args.inner_fmax:g}"
        )
        print(
            f"[OBSM {args.solver} lambda={args.lambda_mix:g} "
            f"inner={inner_label} reparam={args.reparam_interval}] "
            f"outer={outer_iteration:3d} calls={int(np.max(internal)):4d} "
            f"Iperp={diagnostic['i_perp']:.6e} "
            f"barrier={diagnostic['barrier_kcal']:.4f} "
            f"fmax={diagnostic['fmax']:.5f} "
            f"Iarc={diagnostic['i_arc']:.3e}",
            flush=True,
        )

    stop_reason = "evaluation_budget"
    outer = 0
    cycle_detection_enabled = bool(
        args.solver == "fire"
        or (
            args.solver == "bfgs"
            and args.bfgs_state_scope == "paper_local"
        )
    )
    cycle_phase_modulus = max(1, int(args.reparam_interval))
    initial_path_hash = array_sha256(np.asarray(current, dtype=float))
    seen_path_states = {(0, initial_path_hash): 0}
    path_hashes = [initial_path_hash]
    try:
        record(outer)
        while True:
            internal_counts = counter_arrays(evaluators)["joint_evaluations"][1:-1]
            if int(np.max(internal_counts)) >= int(args.budget):
                break
            outer += 1
            reference_path = current.copy()
            tangents = central_tangents(reference_path)
            if ffprecon_bank is not None and not ffprecon_bank.has_dense_hessians and (
                args.bfgs_state_scope == "paper_local"
                or not all(
                    state.initialized for state in persistent_bfgs_states
                )
            ):
                raw_before_h0 = int(backend.raw_joint_evaluations)
                ffprecon_bank.build(reference_path)
                if int(backend.raw_joint_evaluations) != raw_before_h0:
                    raise RuntimeError(
                        "Bonded force-field H0 construction called OpenMM"
                    )
            minimizers = reference_path.copy()
            results: list[InnerResult] = []
            for local_index, image_index in enumerate(range(1, current.shape[0] - 1)):
                bfgs_state = None
                if args.solver == "bfgs":
                    if args.bfgs_state_scope == "curvature_carry":
                        bfgs_state = persistent_bfgs_states[local_index]
                    else:
                        bfgs_state = PersistentBFGSState()
                        bfgs_state_archive[local_index].append(bfgs_state)
                result = minimize_on_frozen_hyperplane(
                    reference=reference_path[image_index],
                    tangent=tangents[local_index],
                    evaluator=evaluators[image_index],
                    solver=args.solver,
                    max_steps=(
                        args.inner_steps
                        if args.inner_mode == "few_steps"
                        else args.max_inner_steps
                    ),
                    inner_fmax=args.inner_fmax,
                    wolfe_c1=args.wolfe_c1,
                    wolfe_c2=args.wolfe_c2,
                    line_search_maxiter=args.line_search_maxiter,
                    bfgs_h0_mode=args.bfgs_h0,
                    hessian_fd_step=args.hessian_fd_step,
                    bfgs_h0_matrix=(
                        ffprecon_bank.dense_hessian(local_index)
                        if ffprecon_bank is not None
                        else None
                    ),
                    bfgs_line_search_direction=args.bfgs_line_search_direction,
                    bfgs_line_search_fallback=args.bfgs_line_search_fallback,
                    armijo_maxiter=args.armijo_maxiter,
                    armijo_shrink=args.armijo_shrink,
                    fire_mass=args.fire_mass,
                    fire_beta=args.fire_beta,
                    fire_dt=args.fire_dt,
                    reserve_evaluations=1,
                    bfgs_state=bfgs_state,
                    inner_stop_norm=args.inner_stop_norm,
                    inner_mode=args.inner_mode,
                )
                minimizers[image_index] = result.position
                results.append(result)

            last_inner_summary = {
                "accepted": int(sum(item.accepted_steps for item in results)),
                "rejected": int(sum(item.rejected_trials for item in results)),
                "converged": int(sum(item.converged for item in results)),
                "energy_nonincrease": int(
                    sum(item.energy <= item.initial_energy + 1e-12 for item in results)
                ),
                "reparameterized": False,
                "stop_reasons": {
                    reason: int(
                        sum(item.stop_reason == reason for item in results)
                    )
                    for reason in sorted(
                        {item.stop_reason for item in results}
                    )
                },
            }
            numerical_failure = any(
                item.stop_reason.startswith("numerical_failure_")
                for item in results
            )
            if numerical_failure:
                stop_reason = "inner_numerical_failure"
                record(outer)
                break
            if (
                sum(item.joint_evaluations_used for item in results) == 0
                and sum(item.accepted_steps for item in results) == 0
                and sum(item.rejected_trials for item in results) == 0
                and sum(item.converged for item in results)
                < current.shape[0] - 2
            ):
                stop_reason = "evaluation_budget_reserve_no_inner_progress"
                break
            if (
                args.inner_mode == "converge"
                and last_inner_summary["converged"] < current.shape[0] - 2
            ):
                # "Minimize until convergence" is only a valid OBSM outer
                # update when every independent hyperplane subproblem has
                # actually met the declared convergence test.  Do not mix a
                # collection of failed/partial subminimizations.
                stop_reason = "inner_subminimization_not_converged"
                record(outer)
                break
            if (
                args.inner_mode == "few_steps"
                and not obsm_subminimization_batch_complete(
                    results,
                    expected_images=current.shape[0] - 2,
                    inner_mode=args.inner_mode,
                    requested_steps=args.inner_steps,
                )
            ):
                # A line-search failure or other early solver stop is not the
                # requested fixed-n_inner experiment.  Preserve every spent
                # call, but do not mix a partially completed batch.
                if (
                    args.solver == "bfgs"
                    and args.bfgs_state_scope == "curvature_carry"
                    and args.bfgs_failure_policy == "restart_curvature"
                ):
                    bfgs_restart_count += 1
                    consecutive_bfgs_restarts += 1
                    record(outer)
                    if consecutive_bfgs_restarts > int(
                        args.bfgs_max_consecutive_restarts
                    ):
                        stop_reason = (
                            "fixed_inner_steps_not_completed_after_"
                            "maximum_curvature_restarts"
                        )
                        break
                    persistent_bfgs_states = [
                        PersistentBFGSState()
                        for _ in range(initial_path.shape[0] - 2)
                    ]
                    for local_index, state in enumerate(
                        persistent_bfgs_states
                    ):
                        bfgs_state_archive[local_index].append(state)
                    continue
                stop_reason = "fixed_inner_steps_not_completed"
                record(outer)
                break

            consecutive_bfgs_restarts = 0

            mixed = reference_path.copy()
            mixed[1:-1] = (
                (1.0 - float(args.lambda_mix)) * reference_path[1:-1]
                + float(args.lambda_mix) * minimizers[1:-1]
            )
            do_reparam = (
                int(args.reparam_interval) > 0
                and outer % int(args.reparam_interval) == 0
            )
            try:
                current = (
                    cubic_arclength_reparameterize(mixed)
                    if do_reparam
                    else mixed
                )
            except (
                FloatingPointError,
                ValueError,
                np.linalg.LinAlgError,
            ) as exc:
                current = reference_path
                stop_reason = (
                    f"reparameterization_numerical_failure_"
                    f"{type(exc).__name__}: {exc}"
                )
                record(outer)
                break
            current[0] = initial_path[0]
            current[-1] = initial_path[-1]
            last_inner_summary["reparameterized"] = bool(do_reparam)
            record(outer)
            if cycle_detection_enabled:
                current_path_hash = array_sha256(
                    np.asarray(current, dtype=float)
                )
                state_key = (
                    int(outer) % cycle_phase_modulus,
                    current_path_hash,
                )
                if state_key in seen_path_states:
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
                history["physical_fmax"][-1] <= float(args.outer_fmax)
                and history["inner_converged_images"][-1] == current.shape[0] - 2
            ):
                stop_reason = "outer_and_inner_fmax"
                break
            if outer >= int(args.max_outer_iterations):
                stop_reason = "outer_iteration_limit"
                break
    except EvaluationBudgetReached:
        stop_reason = "evaluation_budget"

    config = configuration_payload(args)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:12]
    tag = (
        f"{output_namespace}_{args.run_label}_{args.solver}_lam{args.lambda_mix:g}_"
        f"{'inner'+str(args.inner_steps) if args.inner_mode == 'few_steps' else 'innerconv'}_"
        f"r{args.reparam_interval}_"
        f"nocap_cfg{config_hash}_budget{args.budget}"
    ).replace(".", "p")
    output = root / "runs" / output_namespace / f"{tag}.npz"
    arrays = {name: np.asarray(values) for name, values in history.items()}
    final_call = int(history["calls_per_image"][-1])
    if not stored_path_calls or stored_path_calls[-1] != final_call:
        stored_paths.append(current.copy())
        stored_path_calls.append(final_call)
    atomic_save_npz(
        output,
        method=np.array("OBSM"),
        paper_doi=np.array(PAPER_DOI),
        paper_arxiv=np.array(PAPER_ARXIV),
        objective_definition=np.array(OBSM_OBJECTIVE),
        tangent_definition=np.array(TANGENT_DEFINITION),
        reparameterization_definition=np.array(REPARAMETERIZATION),
        budget_boundary_policy=np.array(BUDGET_BOUNDARY_POLICY),
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
        bfgs_h0_initializations_per_internal_image=np.asarray(
            [
                sum(state.h0_initializations for state in image_states)
                for image_states in bfgs_state_archive
            ],
            dtype=int,
        ),
        bfgs_h0_joint_evaluations_per_internal_image=np.asarray(
            [
                sum(state.h0_joint_evaluations for state in image_states)
                for image_states in bfgs_state_archive
            ],
            dtype=int,
        ),
        bfgs_accepted_updates_per_internal_image=np.asarray(
            [
                sum(state.accepted_updates for state in image_states)
                for image_states in bfgs_state_archive
            ],
            dtype=int,
        ),
        bfgs_failed_updates_per_internal_image=np.asarray(
            [
                sum(state.failed_updates for state in image_states)
                for image_states in bfgs_state_archive
            ],
            dtype=int,
        ),
        bfgs_curvature_restarts_total=np.array(bfgs_restart_count, dtype=int),
        **arrays,
    )
    audit = audit_counter_consistency(backend, evaluators)
    audit.update(
        {
            "result_file": str(output),
            "algorithm": "Optimization-Based String Method",
            "paper_doi": PAPER_DOI,
            "paper_arxiv": PAPER_ARXIV,
            "objective_definition": OBSM_OBJECTIVE,
            "contains_ivm_proximal_term": False,
            "contains_global_iperp_acceptance": False,
            "contains_maximum_displacement_cap": False,
            "contains_lbfgs": False,
            "contains_cg": False,
            "strict_original_equations": config["strict_original_equations"],
            "paper_steps_i_to_v_except_h0_and_unspecified_line_search": config[
                "paper_steps_i_to_v_except_h0_and_unspecified_line_search"
            ],
            "output_namespace": output_namespace,
            "tangent_definition": TANGENT_DEFINITION,
            "reparameterization_definition": REPARAMETERIZATION,
            "budget_boundary_policy": BUDGET_BOUNDARY_POLICY,
            "reparameterization_openmm_calls": 0,
            "configuration": config,
            "initial_path_sha256": array_sha256(initial_path),
            "stop_reason": stop_reason,
            "final_inner_stop_reason_counts": last_inner_summary[
                "stop_reasons"
            ],
            "final_call": int(history["calls_per_image"][-1]),
            "bfgs_h0_initializations_per_internal_image": [
                int(sum(state.h0_initializations for state in image_states))
                for image_states in bfgs_state_archive
            ],
            "bfgs_h0_joint_evaluations_per_internal_image": [
                int(sum(state.h0_joint_evaluations for state in image_states))
                for image_states in bfgs_state_archive
            ],
            "bfgs_accepted_updates_per_internal_image": [
                int(sum(state.accepted_updates for state in image_states))
                for image_states in bfgs_state_archive
            ],
            "bfgs_failed_updates_per_internal_image": [
                int(sum(state.failed_updates for state in image_states))
                for image_states in bfgs_state_archive
            ],
            "bfgs_curvature_restarts_total": int(bfgs_restart_count),
            "bfgs_h0_initialized_exactly_once_per_internal_image": (
                bool(
                    args.solver != "bfgs"
                    or (
                        args.bfgs_state_scope == "curvature_carry"
                        and all(
                            sum(
                                state.h0_initializations
                                for state in image_states
                            )
                            == 1
                            for image_states in bfgs_state_archive
                        )
                    )
                )
            ),
            "all_inner_minimizers_nonincreasing_bare_energy": bool(
                np.all(
                    np.asarray(
                        history["inner_bare_energy_nonincrease_images"], dtype=int
                    )
                    == current.shape[0] - 2
                )
            ),
        }
    )
    write_json(output.with_suffix(".audit.json"), audit)
    print(output, flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--initial-path-file",
        default="inputs/common_initial_path_equidistant.npz",
    )
    parser.add_argument("--run-label", default="tuning")
    parser.add_argument("--output-namespace", default="obsm_strict_original")
    parser.add_argument("--solver", choices=("bfgs", "fire"), default="bfgs")
    parser.add_argument("--lambda-mix", type=float, default=0.25)
    parser.add_argument(
        "--inner-mode",
        choices=("converge", "few_steps"),
        default="converge",
        help=(
            "Paper-authorized subminimization interpretation: converge to the "
            "projected-force tolerance, or perform a fixed few steps."
        ),
    )
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument(
        "--max-inner-steps",
        type=int,
        default=10000,
        help="Untuned safety ceiling used only in converge mode; budget usually binds.",
    )
    parser.add_argument("--reparam-interval", type=int, default=5)
    parser.add_argument("--inner-fmax", type=float, default=1e-3)
    parser.add_argument(
        "--inner-stop-norm",
        choices=("l2", "max_atom"),
        default="l2",
    )
    parser.add_argument("--outer-fmax", type=float, default=1e-3)
    parser.add_argument("--wolfe-c1", type=float, default=1e-4)
    parser.add_argument("--wolfe-c2", type=float, default=0.9)
    parser.add_argument("--line-search-maxiter", type=int, default=10)
    parser.add_argument(
        "--bfgs-h0",
        choices=("identity", "finite_difference", "bonded_forcefield"),
        default="identity",
    )
    parser.add_argument("--hessian-fd-step", type=float, default=1e-4)
    parser.add_argument(
        "--bfgs-bonded-h0-cache",
        default="",
        help=(
            "Optional call-free dense bonded H0 cache for curvature_carry; "
            "the embedded initial-path hash must match."
        ),
    )
    parser.add_argument(
        "--bfgs-line-search-direction",
        choices=("paper_p", "projected_p"),
        default="paper_p",
    )
    parser.add_argument(
        "--bfgs-line-search-fallback",
        choices=("stop", "armijo"),
        default="stop",
        help=(
            "Fallback used only when strong Wolfe returns no step; Armijo "
            "uses the same declared BFGS direction and charges every query."
        ),
    )
    parser.add_argument("--armijo-maxiter", type=int, default=20)
    parser.add_argument("--armijo-shrink", type=float, default=0.5)
    parser.add_argument(
        "--bfgs-state-scope",
        choices=("paper_local", "curvature_carry"),
        default="paper_local",
        help=(
            "paper_local initializes H0 once for each new frozen-hyperplane "
            "BFGS solve; curvature_carry is a disclosed cross-outer warm-start "
            "sensitivity because the paper does not specify this choice."
        ),
    )
    parser.add_argument(
        "--bfgs-failure-policy",
        choices=("stop", "restart_curvature"),
        default="stop",
        help=(
            "On an incomplete fixed-step batch, either stop or retain the "
            "unmoved string and charged calls while restarting every carried "
            "BFGS state from the declared H0."
        ),
    )
    parser.add_argument(
        "--bfgs-max-consecutive-restarts",
        type=int,
        default=3,
        help="Safety limit for repeated curvature restarts at one unmoved string.",
    )
    parser.add_argument("--fire-mass", type=float, default=1.0)
    parser.add_argument("--fire-beta", type=float, default=0.1)
    parser.add_argument("--fire-dt", type=float, default=0.02)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--max-outer-iterations", type=int, default=10000)
    parser.add_argument("--save-all-paths", action="store_true")
    parser.add_argument("--save-all-until", type=int, default=0)
    parser.add_argument("--path-interval", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.lambda_mix < 1.0:
        raise ValueError("The OBSM paper requires mixing lambda in the open interval (0, 1)")
    if args.inner_steps <= 0 or args.max_inner_steps <= 0:
        raise ValueError("Inner iteration limits must be positive")
    if args.bfgs_max_consecutive_restarts < 0:
        raise ValueError("BFGS restart limit must be nonnegative")
    if args.armijo_maxiter <= 0 or not 0.0 < args.armijo_shrink < 1.0:
        raise ValueError("Armijo requires maxiter>0 and 0<shrink<1")
    if args.reparam_interval < 0:
        raise ValueError("reparam_interval must be nonnegative; zero means never")
    if args.fire_mass <= 0.0 or args.fire_beta < 0.0 or args.fire_dt <= 0.0:
        raise ValueError("FIRE requires mass>0, beta>=0, and dt>0")
    if not 0.0 < args.wolfe_c1 < args.wolfe_c2 < 1.0:
        raise ValueError("Strong-Wolfe constants require 0<c1<c2<1")
    if args.hessian_fd_step <= 0.0:
        raise ValueError("Hessian finite-difference step must be positive")
    run(args)


if __name__ == "__main__":
    main()
