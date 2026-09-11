from pathlib import Path

import numpy as np
import pytest

from ala2d_development_oracles import CalibratedSyntheticNoiseOracle
from ala2d_oracles import OracleError


CAMPAIGN = Path(__file__).resolve().parents[1]


def _path() -> np.ndarray:
    with np.load(
        CAMPAIGN / "inputs/reference/MFEP_INITIAL_PATH_common.npz",
        allow_pickle=False,
    ) as loaded:
        return np.asarray(loaded["full_path_init"], dtype=np.float64)


def _oracle(master_seed: int = 101, noise_scale: float = 1.0):
    return CalibratedSyntheticNoiseOracle(
        fes_npz=CAMPAIGN / "inputs/reference/fes_XYZ.npz",
        metric_bank_npz=CAMPAIGN / "inputs/reference/current_ivm.npz",
        calibration_npz=CAMPAIGN
        / "inputs/noise_calibration/r1_v13_confirmed_block_noise.npz",
        calibration_provenance_json=CAMPAIGN
        / "inputs/noise_calibration/r1_v13_confirmed_block_noise.provenance.json",
        master_seed=master_seed,
        noise_scale=noise_scale,
    )


def test_calibration_is_complete_and_block_residuals_are_centered():
    path = CAMPAIGN / "inputs/noise_calibration/r1_v13_confirmed_block_noise.npz"
    with np.load(path, allow_pickle=False) as loaded:
        assert not bool(np.asarray(loaded["formal_evidence"]).item())
        assert int(np.asarray(loaded["source_repeat_count"]).item()) == 3
        assert int(np.asarray(loaded["source_paid_band_count"]).item()) == 300
        assert int(np.asarray(loaded["source_image_evaluation_count"]).item()) == 6000
        gradient_residual = np.asarray(loaded["gradient_block_residual"])
        metric_residual = np.asarray(loaded["metric_block_residual"])
    np.testing.assert_allclose(np.mean(gradient_residual, axis=1), 0.0, atol=2e-15)
    np.testing.assert_allclose(np.mean(metric_residual, axis=1), 0.0, atol=2e-13)


def test_fixed_seed_and_band_are_bitwise_reproducible():
    first = _oracle(master_seed=401).evaluate_band(_path(), band_id=17)
    second = _oracle(master_seed=401).evaluate_band(_path(), band_id=17)
    assert np.array_equal(first.grad_F, second.grad_F)
    assert np.array_equal(first.M, second.M)
    assert np.array_equal(first.grad_F_block_means, second.grad_F_block_means)
    assert first.metadata == second.metadata


def test_band_id_and_master_seed_change_the_noise_draw():
    path = _path()
    reference = _oracle(master_seed=401).evaluate_band(path, band_id=17)
    changed_band = _oracle(master_seed=401).evaluate_band(path, band_id=18)
    changed_seed = _oracle(master_seed=402).evaluate_band(path, band_id=17)
    assert not np.array_equal(reference.grad_F, changed_band.grad_F)
    assert not np.array_equal(reference.grad_F, changed_seed.grad_F)


def test_band_has_twenty_mean_force_query_equivalents_and_safe_metrics():
    oracle = _oracle(master_seed=701)
    band = oracle.evaluate_band(_path(), band_id=3)
    assert band.centers.shape == (20, 2)
    assert band.grad_F.shape == (20, 2)
    assert band.grad_F_block_means.shape == (20, 10, 2)
    assert band.M.shape == (20, 2, 2)
    assert band.M_block_means.shape == (20, 10, 2, 2)
    assert band.nof == 20
    assert oracle.call_count == 20
    assert oracle.formal_evidence is False
    assert band.metadata["formal_evidence"] is False
    assert band.metadata["query_equivalents_only"] is True
    assert np.all(band.frame_counts == 2501)
    assert np.all(band.discarded_frame_counts == 125)
    retained = band.frame_counts - band.discarded_frame_counts
    quotient = retained // 10
    remainder = retained % 10
    counts = quotient[:, None] + (np.arange(10)[None, :] < remainder[:, None])
    np.testing.assert_allclose(
        np.einsum("nk,nki->ni", counts, band.grad_F_block_means)
        / retained[:, None],
        band.grad_F,
        atol=2e-15,
    )
    assert float(np.min(np.linalg.eigvalsh(band.M))) > 1e-8
    assert float(np.min(np.linalg.eigvalsh(band.M_block_means))) > 1e-8
    np.testing.assert_allclose(
        np.einsum("nk,nkij->nij", counts, band.M_block_means)
        / retained[:, None, None],
        band.M,
        rtol=1e-14,
        atol=1e-12,
    )


def test_zero_noise_recovers_the_deterministic_surrogate_exactly():
    oracle = _oracle(master_seed=991, noise_scale=0.0)
    path = _path()
    band = oracle.evaluate_band(path, band_id=8)
    expected_gradient = oracle.base._gradient(path[1:-1])
    expected_metric = oracle.base._metric(path[1:-1])
    np.testing.assert_array_equal(band.grad_F, expected_gradient)
    np.testing.assert_allclose(band.M, expected_metric, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(
        band.grad_F_block_means,
        np.repeat(expected_gradient[:, None, :], 10, axis=1),
    )


def test_periodic_equivalent_paths_draw_the_same_local_residuals():
    oracle_a = _oracle(master_seed=811)
    oracle_b = _oracle(master_seed=811)
    path = _path()
    shifted = path + 2.0 * np.pi
    first = oracle_a.evaluate_band(path, band_id=5)
    second = oracle_b.evaluate_band(shifted, band_id=5)
    np.testing.assert_allclose(first.grad_F, second.grad_F, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(first.M, second.M, rtol=1e-13, atol=1e-13)
    assert (
        first.metadata["selected_calibration_indices"]
        == second.metadata["selected_calibration_indices"]
    )


def test_out_of_support_path_fails_closed_before_counting_queries():
    oracle = _oracle(master_seed=812)
    path = _path()
    path[1:-1] = 0.0
    with pytest.raises(OracleError, match="outside.*support"):
        oracle.evaluate_band(path, band_id=6)
    assert oracle.call_count == 0


@pytest.mark.parametrize(
    "keyword",
    [
        "neighbor_bandwidth_rad",
        "maximum_nearest_calibration_distance_rad",
        "maximum_farthest_neighbor_distance_rad",
        "noise_scale",
    ],
)
def test_nonfinite_numeric_control_is_rejected(keyword):
    arguments = {
        "fes_npz": CAMPAIGN / "inputs/reference/fes_XYZ.npz",
        "metric_bank_npz": CAMPAIGN / "inputs/reference/current_ivm.npz",
        "calibration_npz": CAMPAIGN
        / "inputs/noise_calibration/r1_v13_confirmed_block_noise.npz",
        "calibration_provenance_json": CAMPAIGN
        / "inputs/noise_calibration/r1_v13_confirmed_block_noise.provenance.json",
        "master_seed": 813,
        keyword: float("nan"),
    }
    with pytest.raises(ValueError):
        CalibratedSyntheticNoiseOracle(**arguments)
