import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from postprocess_geometric_action import discrete_action, central_tangents, local_path_weights


class GeometricActionTest(unittest.TestCase):
    def test_constant_gradient_exact_polygon_length(self):
        x = np.array([[0., 0.], [1., 0.], [1., 2.], [3., 2.]])
        f = np.tile([3., 4.], (4, 1))
        s, end, norms, lengths = discrete_action(x, f)
        self.assertAlmostEqual(s, 25.)
        self.assertAlmostEqual(s, np.dot(local_path_weights(x), norms[1:-1])+end)

    def test_full_gradient_not_normal_projection(self):
        x = np.array([[0., 0.], [1., 0.], [3., 0.]])
        f = np.tile([2., 0.], (3, 1))
        self.assertAlmostEqual(discrete_action(x, f)[0], 6.)
        np.testing.assert_array_equal(central_tangents(x), [[1., 0.]])

    def test_endpoint_and_reversal(self):
        x = np.array([[0., 0.], [1., 0.], [3., 0.]])
        f = np.array([[2., 0.], [0., 0.], [4., 0.]])
        a = discrete_action(x, f)
        self.assertAlmostEqual(a[0], 5.)
        self.assertAlmostEqual(a[1], 5.)
        self.assertAlmostEqual(a[0], discrete_action(x[::-1], -f[::-1])[0])

    def test_invalid_input_rejected(self):
        x = np.zeros((3, 2))
        with self.assertRaises(ValueError):
            discrete_action(x, np.ones_like(x))


if __name__ == '__main__':
    unittest.main()
