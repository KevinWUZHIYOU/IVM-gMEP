"""Regression checks for spline-free V9 spring projection."""

from __future__ import annotations

import inspect

import numpy as np

import run_ivm_v9_spring_projected_production_20260730 as v9
from chignolin_core import equidistance_diagnostics


def test_spring_projection_reaches_equal_spacing_and_fixes_endpoints():
    parameter = np.array([0.0, 0.03, 0.21, 0.58, 0.86, 1.0])
    path = np.column_stack(
        [
            parameter,
            0.25 * np.sin(np.pi * parameter),
            np.zeros_like(parameter),
        ]
    )
    projected, residual = v9.central_spring_force_equilibrium(path)
    diagnostic = equidistance_diagnostics(projected)
    assert np.array_equal(projected[[0, -1]], path[[0, -1]])
    assert residual <= v9.SPRING_PROJECTION_TOLERANCE
    assert diagnostic["segment_length_cv"] <= 1e-8


def test_projection_source_contains_no_spline_or_openmm_call():
    source = inspect.getsource(v9.central_spring_force_equilibrium)
    assert "CubicHermiteSpline" not in source
    assert "OpenMMBackend" not in source
    assert "evaluate_path" not in source
    assert "spring_reparameterize" in source


def test_projection_rule_is_fixed_and_alpha_independent():
    signature = inspect.signature(v9.central_spring_force_equilibrium)
    assert "alpha" not in signature.parameters
    assert v9.SPRING_PROJECTION_DTAU == 0.5
    assert v9.SPRING_PROJECTION_MAX_ITER == 20000
    assert v9.SPRING_PROJECTION_TOLERANCE == 1e-9


if __name__ == "__main__":
    test_spring_projection_reaches_equal_spacing_and_fixes_endpoints()
    test_projection_source_contains_no_spline_or_openmm_call()
    test_projection_rule_is_fixed_and_alpha_independent()
    print("3 V9 spline-free spring projection tests passed")
