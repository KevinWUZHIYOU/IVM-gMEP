import importlib.util
from pathlib import Path
import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "freeze_bb_irivm_nf8982_0k_section.py"
spec = importlib.util.spec_from_file_location("freeze_section", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_grid_and_basis_are_the_registered_geometry():
    assert module.N_GRID.shape == (28,)
    assert module.N_GRID[0] == -0.10
    assert np.isclose(module.N_GRID[-1], 0.44)
    s = np.linspace(0.0, 1.0, module.POINTS)
    center = np.zeros((module.POINTS, 4, 3))
    center[:, 0, 0] = s
    initial = center.copy()
    initial[:, 1, 1] = np.sin(np.pi * s)
    tangent, normal = module.basis(center, initial)
    assert tangent.shape == normal.shape == center.shape
    flat_t = tangent.reshape(module.POINTS, -1)
    flat_n = normal.reshape(module.POINTS, -1)
    assert np.allclose(np.sum(flat_t * flat_n, axis=1), 0.0, atol=1e-12)
    assert np.allclose(np.linalg.norm(flat_n, axis=1), 1.0)


def test_resampling_is_uniform_in_cartesian_arc_length():
    path = np.zeros((3, 2, 3))
    path[:, 0, 0] = [0.0, 1.0, 3.0]
    target = np.linspace(0.0, 1.0, 7)
    result = module.resample(path, target)
    assert np.allclose(result[:, 0, 0], np.linspace(0.0, 3.0, 7))
