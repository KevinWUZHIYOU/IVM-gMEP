#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess_ala_geometric_action import (  # noqa: E402
    central_tangents,
    discrete_action,
    local_path_weights,
)


class DiscreteGeometricActionTest(unittest.TestCase):
    def test_constant_gradient_exact_polygon_length(self):
        path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [3.0, 2.0]])
        forces = np.tile([3.0, 4.0], (4, 1))
        action, endpoint, norms, _ = discrete_action(path, forces)
        self.assertAlmostEqual(action, 25.0)
        self.assertAlmostEqual(
            action, np.dot(local_path_weights(path), norms[1:-1]) + endpoint
        )

    def test_full_gradient_not_normal_projection(self):
        path = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        forces = np.tile([2.0, 0.0], (3, 1))
        self.assertAlmostEqual(discrete_action(path, forces)[0], 6.0)
        np.testing.assert_array_equal(central_tangents(path), [[1.0, 0.0]])

    def test_endpoint_and_reversal(self):
        path = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        forces = np.array([[2.0, 0.0], [0.0, 0.0], [4.0, 0.0]])
        action = discrete_action(path, forces)
        self.assertAlmostEqual(action[0], 5.0)
        self.assertAlmostEqual(action[1], 5.0)
        self.assertAlmostEqual(action[0], discrete_action(path[::-1], -forces[::-1])[0])

    def test_zero_segment_rejected(self):
        path = np.zeros((3, 2))
        with self.assertRaises(ValueError):
            discrete_action(path, np.ones_like(path))


if __name__ == "__main__":
    unittest.main()
