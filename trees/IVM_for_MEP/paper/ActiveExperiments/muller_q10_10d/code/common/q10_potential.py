#!/usr/bin/env python3
"""Fixed-Q, unperturbed 10D Muller-Brown potential and analytic forces.

Column-vector convention:

    z = Q @ x
    U(x) = U_M(z[0], z[1]) + 2 * sum(z[2:] ** 2)
    grad_x U = Q.T @ grad_z U

For a batch stored as row vectors, ``z = x @ Q.T`` and
``grad_x = grad_z @ Q``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


DIMENSION = 10
N_TOTAL_IMAGES = 22
N_INTERNAL_IMAGES = N_TOTAL_IMAGES - 2

D = np.array([-200.0, -100.0, -170.0, 15.0], dtype=float)
A = np.array([-1.0, -1.0, -6.5, 0.7], dtype=float)
B = np.array([0.0, 0.0, 11.0, 0.6], dtype=float)
C = np.array([-10.0, -10.0, -6.5, 0.7], dtype=float)
X = np.array([1.0, 0.0, -0.5, -1.0], dtype=float)
Y = np.array([0.0, 0.5, 1.5, 1.0], dtype=float)

NATURAL_START = np.array(
    [0.623499399, 0.0280377512] + [0.0] * (DIMENSION - 2), dtype=float
)
NATURAL_END = np.array(
    [-0.558223673, 1.44172580] + [0.0] * (DIMENSION - 2), dtype=float
)

ROOT = Path(__file__).resolve().parents[1]
Q_PATH = ROOT / "inputs" / "Q_data.npz"
with np.load(Q_PATH, allow_pickle=False) as _q_data:
    Q = np.asarray(_q_data["Q"], dtype=float)

if Q.shape != (DIMENSION, DIMENSION):
    raise ValueError(f"Q must have shape {(DIMENSION, DIMENSION)}, got {Q.shape}")

TRANSFORMED_START = Q.T @ NATURAL_START
TRANSFORMED_END = Q.T @ NATURAL_END


def natural_energy_gradient(z: np.ndarray) -> tuple[float, np.ndarray]:
    """Return U0(z) and its analytic gradient for one natural-coordinate point."""
    z = np.asarray(z, dtype=float).reshape(DIMENSION)
    if not np.all(np.isfinite(z)):
        raise FloatingPointError("non-finite natural Muller point")
    dx = z[0] - X
    dy = z[1] - Y
    exponent = A * dx * dx + B * dx * dy + C * dy * dy
    if float(np.max(exponent)) > 650.0:
        raise FloatingPointError("Muller exponential overflow guard")
    terms = D * np.exp(exponent)
    energy = float(np.sum(terms) + 2.0 * np.dot(z[2:], z[2:]))
    gradient = np.empty(DIMENSION, dtype=float)
    gradient[0] = np.sum(terms * (2.0 * A * dx + B * dy))
    gradient[1] = np.sum(terms * (B * dx + 2.0 * C * dy))
    gradient[2:] = 4.0 * z[2:]
    if not np.isfinite(energy) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("non-finite natural Muller energy/gradient")
    return energy, gradient


def energy_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Return U(x) and grad_x U for one Q-transformed Cartesian point."""
    x = np.asarray(x, dtype=float).reshape(DIMENSION)
    z = Q @ x
    energy, gradient_z = natural_energy_gradient(z)
    gradient_x = Q.T @ gradient_z
    return energy, np.asarray(gradient_x, dtype=float)


def energy_force(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Return U(x) and physical force -grad_x U."""
    energy, gradient = energy_gradient(x)
    return energy, -gradient


def energy_values(points: np.ndarray) -> np.ndarray:
    """Vectorized U(x) for a batch of row-vector points with shape (N, 10)."""
    points = np.asarray(points, dtype=float)
    single = points.ndim == 1
    if single:
        points = points.reshape(1, DIMENSION)
    if points.ndim != 2 or points.shape[1] != DIMENSION:
        raise ValueError(f"points must have shape (N,{DIMENSION})")
    if not np.all(np.isfinite(points)):
        raise FloatingPointError("non-finite transformed Muller points")
    z = points @ Q.T
    dx = z[:, 0, None] - X[None, :]
    dy = z[:, 1, None] - Y[None, :]
    exponent = A[None, :] * dx * dx + B[None, :] * dx * dy + C[None, :] * dy * dy
    if float(np.max(exponent)) > 650.0:
        raise FloatingPointError("Muller exponential overflow guard")
    values = np.sum(D[None, :] * np.exp(exponent), axis=1)
    values += 2.0 * np.sum(z[:, 2:] ** 2, axis=1)
    return float(values[0]) if single else values


def natural_initial_path(n_images: int = N_TOTAL_IMAGES) -> np.ndarray:
    """Construct the straight initial path before applying the Q transform."""
    return np.linspace(NATURAL_START, NATURAL_END, int(n_images))


def transformed_initial_path(n_images: int = N_TOTAL_IMAGES) -> np.ndarray:
    """Transform the natural z-space initial path using x = Q.T z."""
    return natural_initial_path(n_images) @ Q


try:
    from ase.calculators.calculator import Calculator, all_changes

    class Q10MullerCalculator(Calculator):
        """ASE calculator using ten atoms' x coordinates as the ten DOFs."""

        implemented_properties = ["energy", "forces"]

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.joint_evaluations = 0

        def calculate(self, atoms=None, properties=("energy", "forces"),
                      system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            positions = np.asarray(atoms.get_positions(), dtype=float)
            if positions.shape != (DIMENSION, 3):
                raise ValueError(
                    f"expected {DIMENSION} atoms with 3D ASE positions, got {positions.shape}"
                )
            x = positions[:, 0]
            energy, force = energy_force(x)
            ase_force = np.zeros_like(positions)
            ase_force[:, 0] = force
            self.results["energy"] = float(energy)
            self.results["forces"] = ase_force
            self.joint_evaluations += 1

except ImportError:
    Q10MullerCalculator = None
