#!/usr/bin/env python3
"""Canonical ASE-independent Q-transformed 10D Muller analytic oracle.

The executable definitions used by formal IR-IVM are an exact source
extraction from ``q10_potential.py`` at SHA-256
``fddf4727c12df28d307fc7e74649f49a320cc7035e914815d1815cd6d8fec1aa``.
ASE calculator integration and reporting-only helpers remain in the historical
module and are deliberately outside this production potential source.

Column-vector convention:

    z = Q @ x
    U(x) = U_M(z[0], z[1]) + 2 * sum(z[2:] ** 2)
    grad_x U = Q.T @ grad_z U
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


FORMULA_ORIGIN_PATH = "q10_potential.py"
FORMULA_ORIGIN_SHA256 = (
    "fddf4727c12df28d307fc7e74649f49a320cc7035e914815d1815cd6d8fec1aa"
)

DIMENSION = 10
N_TOTAL_IMAGES = 22

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


def natural_initial_path(n_images: int = N_TOTAL_IMAGES) -> np.ndarray:
    """Construct the straight initial path before applying the Q transform."""
    return np.linspace(NATURAL_START, NATURAL_END, int(n_images))


def transformed_initial_path(n_images: int = N_TOTAL_IMAGES) -> np.ndarray:
    """Transform the natural z-space initial path using x = Q.T z."""
    return natural_initial_path(n_images) @ Q


__all__ = ["energy_gradient", "transformed_initial_path"]
