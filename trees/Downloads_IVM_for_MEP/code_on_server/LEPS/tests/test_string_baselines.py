import unittest

import numpy as np

from leps_core import (
    LEPSBackend,
    ImageEvaluator,
    leps_energy_gradient,
    physical_to_flat,
)
from run_string_baseline import (
    cubic_arclength_reparameterize,
    lbfgs_direction,
    minimize_on_frozen_normal,
)


class TestStringBaselines(unittest.TestCase):
    def test_cubic_reparameterization_preserves_endpoints(self):
        points = np.array(
            [[0.0, 0.0], [0.2, 0.6], [0.8, 0.9], [2.0, 1.0]]
        )
        flat = physical_to_flat(points)
        result = cubic_arclength_reparameterize(flat)
        np.testing.assert_array_equal(result[0], flat[0])
        np.testing.assert_array_equal(result[-1], flat[-1])
        self.assertTrue(np.all(np.isfinite(result)))

    def test_lbfgs_direction_is_descent(self):
        gradient = np.array([2.0, -1.0])
        direction = lbfgs_direction(gradient, [])
        self.assertLess(float(np.dot(direction, gradient)), 0.0)

    def test_lbfgs_frozen_normal_subsolve_decreases_energy(self):
        backend = LEPSBackend()
        evaluator = ImageEvaluator(backend, 1, max_joint_evaluations=50)
        reference = physical_to_flat([1.7, 1.2])
        tangent = physical_to_flat([1.0, 0.0])
        tangent /= np.linalg.norm(tangent)
        before = float(leps_energy_gradient(1.7, 1.2)[0])
        result, accepted = minimize_on_frozen_normal(
            reference,
            tangent,
            evaluator,
            solver="lbfgs",
            inner_steps=5,
            fire_dt=0.01,
            fire_beta=0.1,
            fire_mass=1.0,
            lbfgs_history=5,
            lbfgs_initial_scale=0.01,
            armijo_c1=1e-4,
            backtrack=0.5,
            line_search_maxiter=12,
        )
        after = float(leps_energy_gradient(result[0], result[3])[0])
        self.assertGreater(accepted, 0)
        self.assertLessEqual(after, before)


if __name__ == "__main__":
    unittest.main()
