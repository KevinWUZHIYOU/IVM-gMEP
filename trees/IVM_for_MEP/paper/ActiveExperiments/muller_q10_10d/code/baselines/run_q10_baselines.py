#!/usr/bin/env python3
"""Grid-search the five non-IVM baselines on the fixed-Q 10D Muller case."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize, root

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.mep.neb import NEB, NEBOptimizer

from q10_potential import (
    DIMENSION,
    N_INTERNAL_IMAGES,
    N_TOTAL_IMAGES,
    NATURAL_START,
    Q,
    TRANSFORMED_END,
    TRANSFORMED_START,
    energy_force,
    energy_gradient,
    energy_values,
    natural_energy_gradient,
    transformed_initial_path,
)


METHODS = (
    "original_string",
    "obsm_fire",
    "obsm_lbfgs",
    "static_neb",
    "ode_neb",
)
Q_HASH = hashlib.sha256(np.ascontiguousarray(Q).tobytes()).hexdigest()


class BudgetReached(RuntimeError):
    pass


@dataclass
class Evaluator:
    budget: int
    charged: bool = True

    def __post_init__(self):
        self.calls = 0
        self.cache = {}

    def query(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        key = x.tobytes()
        if key in self.cache:
            energy, gradient = self.cache[key]
            return energy, gradient.copy()
        if self.charged and self.calls >= int(self.budget):
            raise BudgetReached
        energy, gradient = energy_gradient(x)
        self.cache[key] = (energy, gradient.copy())
        if self.charged:
            self.calls += 1
        return energy, gradient.copy()


class EvaluatorCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, evaluator, **kwargs):
        super().__init__(**kwargs)
        self.evaluator = evaluator

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.get_positions(), dtype=float)
        x = positions[:, 0]
        energy, gradient = self.evaluator.query(x)
        force = np.zeros_like(positions)
        force[:, 0] = -gradient
        self.results["energy"] = float(energy)
        self.results["forces"] = force


def make_evaluators(budget):
    return [
        Evaluator(budget=budget, charged=(0 < i < N_TOTAL_IMAGES - 1))
        for i in range(N_TOTAL_IMAGES)
    ]


def common_path():
    return transformed_initial_path(N_TOTAL_IMAGES)


def evaluate_path(path, evaluators):
    energies = np.empty(N_TOTAL_IMAGES, dtype=float)
    gradients = np.empty_like(path)
    for i in range(N_TOTAL_IMAGES):
        energies[i], gradients[i] = evaluators[i].query(path[i])
    return energies, gradients


def improved_tangents(path, energies):
    tangents = np.empty((N_INTERNAL_IMAGES, DIMENSION), dtype=float)
    for j, i in enumerate(range(1, N_TOTAL_IMAGES - 1)):
        forward = path[i + 1] - path[i]
        backward = path[i] - path[i - 1]
        e_prev, e_now, e_next = energies[i - 1:i + 2]
        if e_next > e_now > e_prev:
            tangent = forward.copy()
        elif e_next < e_now < e_prev:
            tangent = backward.copy()
        else:
            de_plus = abs(e_next - e_now)
            de_minus = abs(e_prev - e_now)
            de_max = max(de_plus, de_minus)
            de_min = min(de_plus, de_minus)
            tangent = (
                forward * de_max + backward * de_min
                if e_next > e_prev
                else forward * de_min + backward * de_max
            )
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-14:
            tangent = forward + backward
            norm = float(np.linalg.norm(tangent))
        if norm <= 1e-14:
            raise FloatingPointError("degenerate improved tangent")
        tangents[j] = tangent / norm
    return tangents


def project_normal(vectors, tangents):
    return vectors - np.sum(vectors * tangents, axis=1)[:, None] * tangents


def third_divided_difference(x, y):
    values = np.asarray(y, dtype=float).copy()
    x = np.asarray(x, dtype=float)
    for order in range(1, x.size):
        denominator = x[order:] - x[:-order]
        values = (values[1:] - values[:-1]) / denominator.reshape(
            (-1,) + (1,) * (values.ndim - 1)
        )
    return np.asarray(values[0], dtype=float)


def appendix_spline_sigmas(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    h = np.diff(x)
    if x.ndim != 1 or y.shape[0] != x.size or x.size < 4 or np.any(h <= 0.0):
        raise ValueError("invalid Appendix-A spline nodes")
    count = x.size
    matrix = np.zeros((count, count), dtype=float)
    rhs = np.zeros_like(y, dtype=float)
    matrix[0, 0] = -h[0]
    matrix[0, 1] = h[0]
    rhs[0] = h[0] ** 2 * third_divided_difference(x[:4], y[:4])
    slopes = np.diff(y, axis=0) / h.reshape((-1,) + (1,) * (y.ndim - 1))
    for i in range(1, count - 1):
        matrix[i, i - 1] = h[i - 1]
        matrix[i, i] = 2.0 * (h[i - 1] + h[i])
        matrix[i, i + 1] = h[i]
        rhs[i] = slopes[i] - slopes[i - 1]
    matrix[-1, -2] = h[-1]
    matrix[-1, -1] = -h[-1]
    rhs[-1] = -(h[-1] ** 2) * third_divided_difference(x[-4:], y[-4:])
    sigma = np.linalg.solve(matrix, rhs.reshape(count, -1))
    return sigma.reshape(y.shape)


def appendix_cubic_spline(x, y, targets):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    targets = np.asarray(targets, dtype=float)
    sigma = appendix_spline_sigmas(x, y)
    interval = np.searchsorted(x, targets, side="right") - 1
    interval = np.clip(interval, 0, x.size - 2)
    h = x[interval + 1] - x[interval]
    w = (targets - x[interval]) / h
    wbar = 1.0 - w
    shape = (-1,) + (1,) * (y.ndim - 1)
    wb = w.reshape(shape)
    wbarb = wbar.reshape(shape)
    h2b = (h**2).reshape(shape)
    return (
        wb * y[interval + 1]
        + wbarb * y[interval]
        + h2b * (
            (wb**3 - wb) * sigma[interval + 1]
            + (wbarb**3 - wbarb) * sigma[interval]
        )
    )


def cubic_arclength_reparameterize(path):
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    if np.any(lengths <= 1e-14):
        raise FloatingPointError("collapsed segment in cubic reparameterization")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    cumulative /= cumulative[-1]
    result = appendix_cubic_spline(
        cumulative, path, np.linspace(0.0, 1.0, path.shape[0])
    )
    result[0] = path[0]
    result[-1] = path[-1]
    return np.asarray(result, dtype=float)


def spring_residual(path):
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return float(np.max(np.abs(np.diff(lengths))))


def dense_delta_energy(path, dense_n=500):
    old = np.linspace(0.0, 1.0, path.shape[0])
    new = np.linspace(0.0, 1.0, int(dense_n))
    dense = np.column_stack(
        [np.interp(new, old, path[:, coordinate]) for coordinate in range(DIMENSION)]
    )
    return float(np.max(energy_values(dense)) - energy_gradient(path[0])[0])


def reference_barrier():
    saddle_energies = []
    for guess in ((0.21, 0.29), (-0.82, 0.62)):
        solution = root(
            lambda point: natural_energy_gradient(
                np.array([point[0], point[1]] + [0.0] * 8)
            )[1][:2],
            guess,
        )
        if not solution.success:
            raise RuntimeError("failed to locate Muller saddle")
        z = np.array([solution.x[0], solution.x[1]] + [0.0] * 8)
        saddle_energies.append(natural_energy_gradient(z)[0])
    return float(max(saddle_energies) - natural_energy_gradient(NATURAL_START)[0])


def diagnostics(path, energies, gradients):
    tangents = improved_tangents(path, energies)
    normal_force = project_normal(-gradients[1:-1], tangents)
    s = np.linspace(0.0, 1.0, N_TOTAL_IMAGES)
    phi_s = np.gradient(path, s, axis=0, edge_order=2)
    phi_ss = np.gradient(phi_s, s, axis=0, edge_order=2)
    speed = np.linalg.norm(phi_s, axis=1)
    ds = 1.0 / (N_TOTAL_IMAGES - 1)
    i_perp = float(
        np.sum(np.sum(normal_force**2, axis=1) * speed[1:-1]) * ds
    )
    arc_integrand = np.sum(phi_s * phi_ss, axis=1) ** 2 * speed
    trapz = getattr(np, "trapezoid", np.trapz)
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return {
        "i_perp": i_perp,
        "i_arc": float(trapz(arc_integrand, s)),
        "delta_e": dense_delta_energy(path),
        "f_perp_max": float(np.max(np.linalg.norm(normal_force, axis=1))),
        "segment_cv": float(np.std(segments) / np.mean(segments)),
        "segment_ratio": float(np.max(segments) / np.min(segments)),
        "spring_residual": spring_residual(path),
    }, normal_force


def record_history(path, evaluators, history, paths, normal_forces, store=True):
    energies, gradients = evaluate_path(path, evaluators)
    if not store:
        return energies, gradients
    row, normal = diagnostics(path, energies, gradients)
    internal = evaluators[1:-1]
    row.update({
        "calls_per_image": int(max(e.calls for e in internal)),
        "mean_calls_per_image": float(np.mean([e.calls for e in internal])),
        "total_force_evaluations": int(sum(e.calls for e in internal)),
    })
    history.append(row)
    paths.append(path.copy())
    normal_forces.append(normal.copy())
    return energies, gradients


def normal_basis(tangent):
    basis = null_space(np.asarray(tangent, dtype=float).reshape(1, -1))
    if basis.shape != (DIMENSION, DIMENSION - 1):
        raise FloatingPointError("normal basis construction failed")
    return basis


def minimize_normal_lbfgs(reference, tangent, evaluator, maxiter):
    basis = normal_basis(tangent)

    def objective(u):
        displacement = basis @ u
        energy, gradient = evaluator.query(reference + displacement)
        return energy, basis.T @ gradient

    result = minimize(
        objective,
        np.zeros(DIMENSION - 1),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": int(maxiter), "maxls": 40, "ftol": 1e-15,
                 "gtol": 1e-13, "maxcor": 20},
    )
    return reference + basis @ np.asarray(result.x, dtype=float)


def minimize_normal_fire(reference, tangent, evaluator, maxiter, fire_dt, fire_beta):
    basis = normal_basis(tangent)
    u = np.zeros(DIMENSION - 1, dtype=float)
    velocity = np.zeros_like(u)
    dt0 = float(fire_dt)
    dt = dt0
    beta0 = float(fire_beta)
    beta = beta0
    positive = 0
    for _ in range(int(maxiter)):
        _, gradient = evaluator.query(reference + basis @ u)
        force = -(basis.T @ gradient)
        velocity += dt * force
        power = float(np.dot(velocity, force))
        if power > 0.0:
            positive += 1
            vnorm = float(np.linalg.norm(velocity))
            fnorm = float(np.linalg.norm(force))
            if vnorm > 0.0 and fnorm > 0.0:
                velocity = (1.0 - beta) * velocity + beta * vnorm * force / fnorm
            if positive > 5:
                dt = min(1.1 * dt, 10.0 * dt0)
                beta *= 0.99
        else:
            velocity.fill(0.0)
            dt *= 0.5
            beta = beta0
            positive = 0
        trial = u + dt * velocity
        if not np.all(np.isfinite(trial)) or np.linalg.norm(trial) > 20.0:
            raise FloatingPointError("OBSM-FIRE non-finite trial")
        u = trial
    return reference + basis @ u


def run_obsm(config, budget):
    path = common_path()
    evaluators = make_evaluators(budget)
    history, paths, normal_forces = [], [], []
    record_history(path, evaluators, history, paths, normal_forces)
    diagnostic_interval = int(config.get("diagnostic_interval", 1))
    stop = "outer_limit"
    for outer in range(1, 10001):
        try:
            energies, _ = evaluate_path(path, evaluators)
            tangents = improved_tangents(path, energies)
            proposed = path.copy()
            for i in range(1, N_TOTAL_IMAGES - 1):
                if config["solver"] == "lbfgs":
                    proposed[i] = minimize_normal_lbfgs(
                        path[i], tangents[i - 1], evaluators[i], config["inner"]
                    )
                else:
                    proposed[i] = minimize_normal_fire(
                        path[i], tangents[i - 1], evaluators[i], config["inner"],
                        config["fire_dt"], config["fire_beta"]
                    )
            updated = path.copy()
            lam = float(config["lambda"])
            updated[1:-1] = path[1:-1] + lam * (proposed[1:-1] - path[1:-1])
            if outer % int(config["reparam_interval"]) == 0:
                updated = cubic_arclength_reparameterize(updated)
            movement = float(np.max(np.linalg.norm(updated - path, axis=1)))
            path = updated
            stored = outer % diagnostic_interval == 0
            record_history(
                path, evaluators, history, paths, normal_forces, store=stored
            )
            if movement <= 1e-14:
                if not stored:
                    record_history(
                        path, evaluators, history, paths, normal_forces, store=True
                    )
                stop = "stationary"
                break
        except BudgetReached:
            stop = "budget"
            break
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            stop = "numerical_failure"
            break
    return path, evaluators, history, paths, normal_forces, stop


def run_original_string(config, budget):
    path = common_path()
    evaluators = make_evaluators(budget)
    history, paths, normal_forces = [], [], []
    _, gradients = record_history(path, evaluators, history, paths, normal_forces)
    diagnostic_interval = int(config.get("diagnostic_interval", 1))
    stop = "outer_limit"
    for outer in range(1, 100001):
        updated = path.copy()
        updated[1:-1] -= float(config["dt"]) * gradients[1:-1]
        if outer % int(config["reparam_interval"]) == 0:
            updated = cubic_arclength_reparameterize(updated)
        if not np.all(np.isfinite(updated)) or np.max(np.abs(updated)) > 20.0:
            stop = "numerical_failure"
            break
        movement = float(np.max(np.linalg.norm(updated - path, axis=1)))
        path = updated
        try:
            stored = outer % diagnostic_interval == 0
            _, gradients = record_history(
                path, evaluators, history, paths, normal_forces, store=stored
            )
        except (BudgetReached, FloatingPointError):
            stop = "budget"
            break
        if movement <= 1e-14:
            if not stored:
                record_history(
                    path, evaluators, history, paths, normal_forces, store=True
                )
            stop = "stationary"
            break
    return path, evaluators, history, paths, normal_forces, stop


def point_atoms(x, evaluator):
    positions = np.zeros((DIMENSION, 3), dtype=float)
    positions[:, 0] = np.asarray(x, dtype=float)
    atoms = Atoms("H" * DIMENSION, positions=positions)
    atoms.calc = EvaluatorCalculator(evaluator)
    return atoms


def path_from_images(images):
    return np.asarray([image.get_positions()[:, 0] for image in images], dtype=float)


def run_ase_neb(config, budget, optimizer_method):
    initial = common_path()
    evaluators = make_evaluators(budget)
    images = [point_atoms(initial[i], evaluators[i]) for i in range(N_TOTAL_IMAGES)]
    neb = NEB(
        images,
        k=float(config["k"]),
        climb=False,
        parallel=False,
        method="improvedtangent",
        allow_shared_calculator=False,
    )
    history, paths, normal_forces = [], [], []
    record_history(initial, evaluators, history, paths, normal_forces)
    if optimizer_method == "ode":
        # Canonical ODE-NEB comparison: retain every ASE constructor default.
        # The spring constant k on the NEB object is the only tuned parameter.
        optimizer = NEBOptimizer(neb, method="ode")
    else:
        optimizer = NEBOptimizer(
            neb,
            method="static",
            logfile=None,
            alpha=float(config["alpha"]),
        )

    def observer():
        path = path_from_images(images)
        record_history(path, evaluators, history, paths, normal_forces)

    optimizer.attach(observer, interval=int(config.get("diagnostic_interval", 1)))
    stop = "iteration_limit"
    try:
        converged = optimizer.run(
            fmax=float(config.get("fmax", 1e-14)),
            steps=int(config.get("max_steps", 100000000)),
        )
        stop = "converged" if converged else "iteration_limit"
    except BudgetReached:
        stop = "budget"
    except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError):
        stop = "numerical_failure"
    path = path_from_images(images)
    return path, evaluators, history, paths, normal_forces, stop


def run_static_neb(config, budget):
    return run_ase_neb(config, budget, "static")


def run_ode_neb(config, budget):
    return run_ase_neb(config, budget, "ode")


def configuration_id(method, config):
    payload = json.dumps(
        {"case": "q10_fixed_seed0", "q_sha256": Q_HASH, "method": method, **config},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def trajectory_summary(history, budget):
    if not history:
        return {
            "i_perp_log_auc": math.inf,
            "barrier_log_auc": math.inf,
            "trajectory_score": math.inf,
            "i_perp_at_call50": math.inf,
            "barrier_error_at_call50": math.inf,
        }
    reference = reference_barrier()
    cost = np.asarray([row["total_force_evaluations"] for row in history], dtype=float)
    keep = np.concatenate(([True], np.diff(cost) > 0.0))
    cost = cost[keep]
    i_perp = np.asarray([row["i_perp"] for row in history], dtype=float)[keep]
    barrier_error = np.abs(
        np.asarray([row["delta_e"] for row in history], dtype=float)[keep] - reference
    )
    maximum = max(float(cost[0]), min(float(cost[-1]), float(budget * N_INTERNAL_IMAGES)))
    samples = np.linspace(float(cost[0]), maximum, 301)
    log_i = np.log10(np.maximum(i_perp, 1e-30))
    log_e = np.log10(np.maximum(barrier_error, 1e-14))
    call50 = min(maximum, 50.0 * N_INTERNAL_IMAGES)
    return {
        "i_perp_log_auc": float(np.mean(np.interp(samples, cost, log_i))),
        "barrier_log_auc": float(np.mean(np.interp(samples, cost, log_e))),
        "trajectory_score": float(
            np.mean(np.interp(samples, cost, log_i))
            + np.mean(np.interp(samples, cost, log_e))
            + 0.25 * (log_i[-1] + log_e[-1])
        ),
        "i_perp_at_call50": float(10.0 ** np.interp(call50, cost, log_i)),
        "barrier_error_at_call50": float(10.0 ** np.interp(call50, cost, log_e)),
    }


def save_run(root, method, config, budget, role):
    runner = {
        "original_string": run_original_string,
        "obsm_fire": run_obsm,
        "obsm_lbfgs": run_obsm,
        "static_neb": run_static_neb,
        "ode_neb": run_ode_neb,
    }[method]
    try:
        path, evaluators, history, paths, normal_forces, stop = runner(config, budget)
        finite = bool(history and np.all(np.isfinite(path)))
    except Exception as exc:
        path = common_path()
        evaluators = make_evaluators(budget)
        history, paths, normal_forces = [], [], []
        stop = "failure_{}:{}".format(type(exc).__name__, exc)
        finite = False
    config_id = configuration_id(method, config)
    output_dir = root / "runs" / role / method
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "{}_{}_budget{}.npz".format(method, config_id, budget)
    final = history[-1] if history else {
        "i_perp": math.inf, "i_arc": math.inf, "delta_e": math.inf,
        "f_perp_max": math.inf, "segment_cv": math.inf, "segment_ratio": math.inf,
        "spring_residual": math.inf, "calls_per_image": 0,
        "mean_calls_per_image": 0.0, "total_force_evaluations": 0,
    }
    arrays = {}
    if history:
        for key in history[0]:
            arrays[key] = np.asarray([row[key] for row in history])
    store_paths = role != "screen"
    np.savez_compressed(
        output,
        case=np.asarray("fixed-Q unperturbed 10D Muller-Brown"),
        q_sha256=np.asarray(Q_HASH),
        q_matrix=Q,
        coordinate_convention=np.asarray("z=Q@x; x=Q.T@z"),
        method=np.asarray(method),
        role=np.asarray(role),
        config_json=np.asarray(json.dumps(config, sort_keys=True)),
        config_id=np.asarray(config_id),
        budget_per_image=np.asarray(budget),
        natural_initial_path=np.linspace(NATURAL_START, np.array([-0.558223673, 1.44172580] + [0.0] * 8), N_TOTAL_IMAGES),
        transformed_initial_path=common_path(),
        final_path=path,
        path_history=np.asarray(paths) if store_paths else np.empty((0, N_TOTAL_IMAGES, DIMENSION)),
        normal_force_history=np.asarray(normal_forces) if store_paths else np.empty((0, N_INTERNAL_IMAGES, DIMENSION)),
        stop_reason=np.asarray(stop),
        calls_per_image_final=np.asarray([e.calls for e in evaluators[1:-1]]),
        reference_barrier=np.asarray(reference_barrier()),
        tangent_definition=np.asarray("ASE Henkelman-Jonsson improved tangent"),
        **arrays
    )
    barrier_error = float(abs(final["delta_e"] - reference_barrier()))
    eligible = bool(
        finite and np.isfinite(final["i_perp"])
        and final["segment_ratio"] <= 5.0 and barrier_error <= 2.0
    )
    return {
        "case": "fixed-Q unperturbed 10D Muller-Brown",
        "q_sha256": Q_HASH,
        "method": method,
        "role": role,
        "config_id": config_id,
        "config": config,
        "budget_per_image": budget,
        "output": str(output.relative_to(root)),
        "stop_reason": stop,
        "finite": finite,
        "eligible": eligible,
        "final_calls_per_image": int(final["calls_per_image"]),
        "final_mean_calls_per_image": float(final["mean_calls_per_image"]),
        "final_total_force_evaluations": int(final["total_force_evaluations"]),
        "final_i_perp": float(final["i_perp"]),
        "final_i_arc": float(final["i_arc"]),
        "final_delta_e": float(final["delta_e"]),
        "barrier_error": barrier_error,
        "final_f_perp_max": float(final["f_perp_max"]),
        "final_segment_cv": float(final["segment_cv"]),
        "final_segment_ratio": float(final["segment_ratio"]),
        "final_spring_residual": float(final["spring_residual"]),
        **trajectory_summary(history, budget)
    }


def grid_specs():
    specs = []
    lambdas = (0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
    for lam, inner, interval in itertools.product(
        lambdas, (1, 2, 4, 8, 16), (1, 2, 5)
    ):
        specs.append(("obsm_lbfgs", {
            "solver": "lbfgs", "lambda": lam, "inner": inner,
            "reparam_interval": interval,
        }))
    for lam, inner, dt, beta, interval in itertools.product(
        lambdas, (2, 4, 8, 16, 32),
        (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
        (0.1, 1.0), (1, 5)
    ):
        specs.append(("obsm_fire", {
            "solver": "fire", "lambda": lam, "inner": inner,
            "fire_dt": dt, "fire_beta": beta, "reparam_interval": interval,
        }))
    for dt, interval in itertools.product(
        (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
        (1, 2, 5)
    ):
        specs.append(("original_string", {"dt": dt, "reparam_interval": interval}))
    for k, rtol, c2 in itertools.product(
        (1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0),
        (0.3, 0.1, 0.03, 0.01, 0.003, 0.001),
        (2.0, 10.0),
    ):
        specs.append(("ode_neb", {
            "k": k, "rtol": rtol, "C1": 0.01, "C2": c2,
            "fmax": 1e-14,
        }))
    for k, alpha in itertools.product(
        (0.3, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0),
        (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    ):
        specs.append(("static_neb", {
            "k": k, "alpha": alpha, "optimizer_method": "static",
            "fmax": 1e-14,
        }))
    return specs


def write_summary(root, rows, stem):
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / (stem + ".json")).open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
    if not rows:
        return
    flat = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "config"}
        item["config_json"] = json.dumps(row["config"], sort_keys=True)
        flat.append(item)
    with (report_dir / (stem + ".csv")).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def worker(payload):
    root, method, config, budget, role = payload
    return save_run(Path(root), method, config, budget, role)


def run_payloads(payloads, workers, label):
    rows = []
    if workers == 1:
        for index, payload in enumerate(payloads, 1):
            rows.append(worker(payload))
            if index % 25 == 0 or index == len(payloads):
                print("{} {}/{}".format(label, index, len(payloads)), flush=True)
        return rows
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, payload) for payload in payloads]
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print("{} {}/{}".format(label, index, len(futures)), flush=True)
    return rows


def run_grid(root, budget, workers):
    specs = grid_specs()
    payloads = [(str(root), method, config, budget, "screen") for method, config in specs]
    rows = run_payloads(payloads, workers, "baseline_screen")
    rows.sort(key=lambda row: (row["method"], row["config_id"]))
    write_summary(root, rows, "baseline_screen_summary")
    return rows


def candidate_configs(rows, candidates_per_method):
    selected = []
    for method in METHODS:
        pool = [row for row in rows if row["method"] == method and row["finite"]]
        eligible = [row for row in pool if row["eligible"]]
        pool = eligible or pool
        rankings = (
            sorted(pool, key=lambda row: (row["trajectory_score"], row["final_i_perp"])),
            sorted(pool, key=lambda row: (row["final_i_perp"], row["barrier_error"])),
            sorted(pool, key=lambda row: (row["barrier_error"], row["final_i_perp"])),
            sorted(pool, key=lambda row: (row["i_perp_at_call50"], row["barrier_error_at_call50"])),
        )
        seen = set()
        candidates = []
        rank = 0
        while len(candidates) < int(candidates_per_method):
            added = False
            for ranking in rankings:
                if rank < len(ranking):
                    row = ranking[rank]
                    if row["config_id"] not in seen:
                        seen.add(row["config_id"])
                        candidates.append(row)
                        added = True
                        if len(candidates) >= int(candidates_per_method):
                            break
            if not added and rank >= max(map(len, rankings)):
                break
            rank += 1
        selected.extend(candidates)
    return selected


def run_production(root, budget, workers, candidates_per_method):
    with (root / "reports" / "baseline_screen_summary.json").open() as handle:
        screen = json.load(handle)
    candidates = candidate_configs(screen, candidates_per_method)
    write_summary(root, candidates, "baseline_production_candidates")
    payloads = [
        (str(root), row["method"], row["config"], budget, "production_candidates")
        for row in candidates
    ]
    rows = run_payloads(payloads, workers, "baseline_production")
    rows.sort(key=lambda row: (row["method"], row["trajectory_score"]))
    write_summary(root, rows, "baseline_production_candidate_summary")
    winners = []
    for method in METHODS:
        pool = [row for row in rows if row["method"] == method and row["finite"]]
        eligible = [row for row in pool if row["eligible"]]
        selection_pool = eligible or pool
        winner = min(
            selection_pool,
            key=lambda row: (row["trajectory_score"], row["final_i_perp"], row["barrier_error"]),
        )
        winner = dict(winner)
        winner["selection_status"] = "eligible" if winner["eligible"] else "ineligible"
        winners.append(winner)
    write_summary(root, winners, "baseline_production_summary")
    return winners


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("grid", "production", "all"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--screen-budget", type=int, default=250)
    parser.add_argument("--production-budget", type=int, default=2000)
    parser.add_argument("--candidates-per-method", type=int, default=8)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command in ("grid", "all"):
        run_grid(root, args.screen_budget, args.workers)
    if args.command in ("production", "all"):
        winners = run_production(
            root, args.production_budget, args.workers, args.candidates_per_method
        )
        for row in winners:
            print(row["method"], row["config"], row["final_i_perp"],
                  row["barrier_error"], row["selection_status"])


if __name__ == "__main__":
    main()
