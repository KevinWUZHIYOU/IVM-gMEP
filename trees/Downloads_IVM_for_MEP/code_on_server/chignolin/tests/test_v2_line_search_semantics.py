"""Regression checks for the V2 projected line-search patience."""

from __future__ import annotations

import numpy as np

from chignolin_core import (
    ImageEvaluator,
    central_tangents,
    evaluate_path,
)
from run_ivm_early_block_relock_20260729 import (
    solve_curve_projected_proximal_band as solve_v1,
)
from run_ivm_early_block_relock_v2_20260729 import (
    solve_curve_projected_proximal_band as solve_v2,
)


class FakeBackend:
    def __init__(self, dimension=3):
        self.dimension = dimension
        self.n_atoms = dimension // 3
        self.raw_joint_evaluations = 0
        self.label = "fake"

    def evaluate_joint(self, positions_flat):
        position = np.asarray(positions_flat, dtype=float)
        self.raw_joint_evaluations += 1
        return float(0.5 * np.dot(position, position)), -position.copy()


def path_fixture():
    angles = np.linspace(0.0, 1.2, 5)
    return np.column_stack(
        (
            2.0 + np.cos(angles),
            np.sin(angles),
            np.zeros_like(angles),
        )
    )


def run_solver(solver, **kwargs):
    path = path_fixture()
    backend = FakeBackend()
    evaluators = [
        ImageEvaluator(backend, image_index)
        for image_index in range(len(path))
    ]
    energies, forces = evaluate_path(path, evaluators)
    result = solver(
        path,
        energies,
        forces,
        central_tangents(path),
        evaluators,
        alpha=1.0,
        max_step=0.2,
        max_new_calls=20,
        kkt_rtol=0.01,
        **kwargs,
    )
    return result, backend


def test_v2_default_is_bitwise_identical_to_archived_v1():
    v1, backend_v1 = run_solver(solve_v1)
    v2, backend_v2 = run_solver(solve_v2)
    np.testing.assert_array_equal(v1[0], v2[0])
    for key in v1[-1]:
        np.testing.assert_array_equal(v1[-1][key], v2[-1][key])
    assert backend_v1.raw_joint_evaluations == backend_v2.raw_joint_evaluations


def test_four_rejected_trials_do_not_cap_successful_inner_iterations():
    result, backend = run_solver(
        solve_v2, line_search_trial_limit=4
    )
    report = result[-1]
    assert int(np.max(report["accepted_steps"])) == 7
    assert int(np.max(report["accepted_steps"])) > 4
    assert int(np.max(report["backtracks"])) == 4
    assert int(np.max(result[3])) == 11
    assert backend.raw_joint_evaluations == 5 + 3 * 11


if __name__ == "__main__":
    test_v2_default_is_bitwise_identical_to_archived_v1()
    test_four_rejected_trials_do_not_cap_successful_inner_iterations()
    print("2 V2 line-search regression tests passed")
