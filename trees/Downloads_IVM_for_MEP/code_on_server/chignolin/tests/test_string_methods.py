import numpy as np

from run_obsm import (
    appendix_cubic_spline,
    appendix_spline_sigmas,
    cubic_arclength_reparameterize,
    third_divided_difference,
)
from run_string_family import one_full_force_step


def test_original_string_step_uses_full_unprojected_force():
    path = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]
    )
    forces = np.asarray([[2.0, 3.0], [-4.0, 5.0]])
    updated = one_full_force_step(path, forces, dt=0.25)
    expected = path.copy()
    expected[1:-1] += 0.25 * forces
    assert np.array_equal(updated, expected)
    # In this geometry the x force is purely tangential.  It must survive the
    # unconstrained Original String descent step.
    assert updated[1, 0] == 1.5
    assert updated[2, 0] == 1.0


def test_appendix_cubic_spline_interpolates_every_node():
    x = np.array([0.0, 0.08, 0.25, 0.55, 0.72, 1.0])
    y = np.column_stack((np.sin(2.0 * x), np.cos(3.0 * x), x**2))
    reconstructed = appendix_cubic_spline(x, y, x)
    np.testing.assert_allclose(reconstructed, y, rtol=0.0, atol=2e-15)


def test_appendix_cubic_spline_exactly_reproduces_a_cubic():
    x = np.array([0.0, 0.11, 0.30, 0.62, 0.79, 1.0])
    targets = np.linspace(0.0, 1.0, 101)

    def polynomial(z):
        return np.column_stack(
            (
                1.2 - 0.7 * z + 2.1 * z**2 - 0.9 * z**3,
                -0.4 + 1.5 * z - 0.2 * z**2 + 0.35 * z**3,
            )
        )

    actual = appendix_cubic_spline(x, polynomial(x), targets)
    np.testing.assert_allclose(
        actual, polynomial(targets), rtol=0.0, atol=5e-14
    )


def test_appendix_sigma_system_has_required_boundaries():
    x = np.array([0.0, 0.07, 0.21, 0.50, 0.81, 1.0])
    y = np.column_stack((np.exp(x), np.sin(4.0 * x)))
    sigma = appendix_spline_sigmas(x, y)
    h = np.diff(x)
    np.testing.assert_allclose(
        6.0 * (sigma[1] - sigma[0]) / h[0],
        6.0 * third_divided_difference(x[:4], y[:4]),
        rtol=0.0,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        6.0 * (sigma[-1] - sigma[-2]) / h[-1],
        6.0 * third_divided_difference(x[-4:], y[-4:]),
        rtol=0.0,
        atol=2e-13,
    )


def test_cubic_reparameterization_preserves_endpoints():
    path = np.array(
        [[0.0, 0.0], [0.1, 0.2], [0.4, 0.3], [1.0, 0.0]],
        dtype=float,
    )
    result = cubic_arclength_reparameterize(path)
    np.testing.assert_array_equal(result[0], path[0])
    np.testing.assert_array_equal(result[-1], path[-1])
    assert result.shape == path.shape
