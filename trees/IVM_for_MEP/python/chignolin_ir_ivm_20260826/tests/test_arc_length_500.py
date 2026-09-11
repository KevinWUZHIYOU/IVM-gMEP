#!/usr/bin/env python3
"""OpenMM-free tests for the fixed arc-length diagnostic and plot helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from plot_chignolin_ivm_arc500 import first_passage, pareto_frontier  # noqa: E402
from postprocess_arc_length_500 import (  # noqa: E402
    N_DENSE_POINTS,
    arc_length_resample,
)


class ArcLength500Test(unittest.TestCase):
    def test_resample_has_exact_count_endpoints_and_uniform_spacing(self) -> None:
        path = np.zeros((22, 3), dtype=float)
        path[:, 0] = np.linspace(0.0, 10.0, 22) ** 1.3
        dense, details = arc_length_resample(path, N_DENSE_POINTS)
        self.assertEqual(dense.shape, (500, 3))
        np.testing.assert_array_equal(dense[0], path[0])
        np.testing.assert_array_equal(dense[-1], path[-1])
        spacing = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        np.testing.assert_allclose(spacing, spacing[0], rtol=2e-13, atol=2e-13)
        self.assertAlmostEqual(details["total_arc_length"], path[-1, 0])

    def test_resample_rejects_zero_length_segment(self) -> None:
        path = np.zeros((22, 3), dtype=float)
        path[:, 0] = np.arange(22)
        path[7] = path[6]
        with self.assertRaisesRegex(ValueError, "positive length"):
            arc_length_resample(path)

    def test_first_passage_uses_running_best_and_earliest_cost(self) -> None:
        cost = np.asarray([30, 10, 20, 40])
        metric = np.asarray([3.0, 8.0, 4.0, 6.0])
        self.assertEqual(first_passage(cost, metric, 4.5), 20)
        self.assertIsNone(first_passage(cost, metric, 2.0))

    def test_frontier_keeps_only_strict_improvements(self) -> None:
        value, cost = pareto_frontier(
            np.asarray([10, 20, 30, 40]),
            np.asarray([8.0, 6.0, 7.0, 4.0]),
            0.0,
            10.0,
        )
        np.testing.assert_array_equal(value, [8.0, 6.0, 4.0])
        np.testing.assert_array_equal(cost, [10.0, 20.0, 40.0])


if __name__ == "__main__":
    unittest.main()
