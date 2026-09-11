"""Focused regression tests for the V8 state-only block/spring policy."""

from __future__ import annotations

import numpy as np

from run_ivm_v8_block_spring_production_20260730 import (
    largest_normal_force_block_mask,
    spring_proposal_to_target_relative_defect,
)


def test_largest_normal_force_block_is_unique_and_state_only():
    per_image_l2 = np.array([0.4, 1.7, 1.2, 0.2])
    selected = largest_normal_force_block_mask(per_image_l2)
    assert selected.dtype == np.bool_
    assert selected.tolist() == [False, True, False, False]
    assert np.count_nonzero(selected) == 1


def test_largest_normal_force_block_rejects_invalid_diagnostics():
    invalid_diagnostics = [
        np.array([]),
        np.zeros((2, 2)),
        np.array([0.1, np.nan]),
        np.array([0.1, np.inf]),
    ]
    for bad in invalid_diagnostics:
        try:
            largest_normal_force_block_mask(bad)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for {bad!r}")


def test_relative_defect_spring_is_geometry_only_and_reaches_target():
    # Nonuniform samples of a smooth two-dimensional curve, embedded in the
    # same flat-coordinate representation used by a molecular path.
    parameter = np.array([0.0, 0.03, 0.21, 0.58, 0.86, 1.0])
    path = np.column_stack(
        [
            parameter,
            0.25 * np.sin(np.pi * parameter),
            np.zeros_like(parameter),
        ]
    )
    target = 2.5e-4
    candidate, _, scale, before, after = (
        spring_proposal_to_target_relative_defect(
            path,
            target,
            dtau=0.05,
        )
    )
    assert 0.0 < scale <= 1.0
    assert np.array_equal(candidate[[0, -1]], path[[0, -1]])
    assert (
        after["equidistance_relative_defect"]
        < before["equidistance_relative_defect"]
    )
    assert after["equidistance_relative_defect"] <= target * (1.0 + 1e-8)


def test_relative_defect_spring_is_exact_noop_below_trigger():
    parameter = np.linspace(0.0, 1.0, 6)
    path = np.column_stack(
        [parameter, np.zeros_like(parameter), np.zeros_like(parameter)]
    )
    candidate, residual, scale, before, after = (
        spring_proposal_to_target_relative_defect(path, 1e-8)
    )
    assert np.array_equal(candidate, path)
    assert residual == 0.0
    assert scale == 0.0
    assert before == after


if __name__ == "__main__":
    test_largest_normal_force_block_is_unique_and_state_only()
    test_largest_normal_force_block_rejects_invalid_diagnostics()
    test_relative_defect_spring_is_geometry_only_and_reaches_target()
    test_relative_defect_spring_is_exact_noop_below_trigger()
    print("4 V8 block/spring policy tests passed")
