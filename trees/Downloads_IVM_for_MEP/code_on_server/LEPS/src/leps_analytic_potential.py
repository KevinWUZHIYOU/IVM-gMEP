#!/usr/bin/env python3
"""Canonical ASE-independent analytic LEPS energy-gradient oracle.

The eight numerical constants and three executable functions below are an
exact source extraction from ``src/leps_core.py`` at SHA-256
``a80f0c23252a2e273f4cf77d156c9a26de91b4559f56a4fad264638fa0e57aca``.
The formal IR-IVM experiment needs only this analytic oracle; ASE-dependent
NEB, tangent, calculator, caching, and path-driver code remain in the original
historical module and are deliberately outside this potential source.
"""

from __future__ import annotations

import numpy as np


FORMULA_ORIGIN_PATH = "src/leps_core.py"
FORMULA_ORIGIN_SHA256 = (
    "a80f0c23252a2e273f4cf77d156c9a26de91b4559f56a4fad264638fa0e57aca"
)

A_LEPS = 0.05
B_LEPS = 0.30
C_LEPS = 0.05
D_AB = 4.746
D_BC = 4.746
D_AC = 3.445
R0 = 0.742
MORSE_ALPHA = 1.942


def _q_and_derivative(r, dissociation):
    r = np.asarray(r, dtype=float)
    displacement = r - R0
    exp_one = np.exp(-MORSE_ALPHA * displacement)
    exp_two = exp_one * exp_one
    value = 0.5 * dissociation * (1.5 * exp_two - exp_one)
    derivative = (
        0.5
        * dissociation
        * MORSE_ALPHA
        * (-3.0 * exp_two + exp_one)
    )
    return value, derivative


def _j_and_derivative(r, dissociation):
    r = np.asarray(r, dtype=float)
    displacement = r - R0
    exp_one = np.exp(-MORSE_ALPHA * displacement)
    exp_two = exp_one * exp_one
    value = 0.25 * dissociation * (exp_two - 6.0 * exp_one)
    derivative = (
        0.25
        * dissociation
        * MORSE_ALPHA
        * (-2.0 * exp_two + 6.0 * exp_one)
    )
    return value, derivative


def leps_energy_gradient(r_ab, r_bc):
    """Return the historical LEPS energy and its two-coordinate gradient."""
    r_ab = np.asarray(r_ab, dtype=float)
    r_bc = np.asarray(r_bc, dtype=float)
    r_ac = r_ab + r_bc

    q_ab, dq_ab = _q_and_derivative(r_ab, D_AB)
    q_bc, dq_bc = _q_and_derivative(r_bc, D_BC)
    q_ac, dq_ac = _q_and_derivative(r_ac, D_AC)
    j_ab, dj_ab = _j_and_derivative(r_ab, D_AB)
    j_bc, dj_bc = _j_and_derivative(r_bc, D_BC)
    j_ac, dj_ac = _j_and_derivative(r_ac, D_AC)

    pa = 1.0 + A_LEPS
    pb = 1.0 + B_LEPS
    pc = 1.0 + C_LEPS

    term_q = q_ab / pa + q_bc / pb + q_ac / pc
    radicand = (
        j_ab**2 / pa**2
        + j_bc**2 / pb**2
        + j_ac**2 / pc**2
        - j_ab * j_bc / (pa * pb)
        - j_bc * j_ac / (pb * pc)
        - j_ab * j_ac / (pa * pc)
    )
    radicand = np.maximum(radicand, 0.0)
    root = np.sqrt(radicand)
    energy = term_q - root

    d_radicand_ab = (
        2.0 * j_ab * dj_ab / pa**2
        + 2.0 * j_ac * dj_ac / pc**2
        - dj_ab * j_bc / (pa * pb)
        - j_bc * dj_ac / (pb * pc)
        - (dj_ab * j_ac + j_ab * dj_ac) / (pa * pc)
    )
    d_radicand_bc = (
        2.0 * j_bc * dj_bc / pb**2
        + 2.0 * j_ac * dj_ac / pc**2
        - j_ab * dj_bc / (pa * pb)
        - (dj_bc * j_ac + j_bc * dj_ac) / (pb * pc)
        - j_ab * dj_ac / (pa * pc)
    )
    safe_root = np.maximum(root, 1e-300)
    grad_ab = dq_ab / pa + dq_ac / pc - 0.5 * d_radicand_ab / safe_root
    grad_bc = dq_bc / pb + dq_ac / pc - 0.5 * d_radicand_bc / safe_root
    gradient = np.stack((grad_ab, grad_bc), axis=-1)
    return energy, gradient


__all__ = ["leps_energy_gradient"]
