import unittest
from pathlib import Path

import numpy as np
import torch

from continuous_paths import (
    FixedEndpointPathNet,
    arclength_parameters_for_fractions,
    dense_linear_metrics,
    dense_neural_metrics,
    equal_arclength_sample,
    fit_path_targets,
    load_parameter_vector,
    model_coordinates,
    normalized_autograd_tangent,
    parameter_vector,
    resample_polyline_equal_arclength,
    stratified_jitter_fractions,
)
from leps_core import (
    LEPSBackend,
    N_IMAGES,
    PRODUCT,
    REACTANT,
    counter_arrays,
    load_common_path,
    make_evaluators,
    physical_to_flat,
    leps_energy_gradient,
)
from run_nn_ivm_theta_al_adaptive import audited_joint_energy_force
from torch_leps import torch_leps_energy_force


ROOT = Path(__file__).resolve().parents[1]


class TestContinuousPath(unittest.TestCase):
    def test_zero_network_is_exact_common_initial_path(self):
        model = FixedEndpointPathNet(width=16, depth=2, seed=5)
        s_values = np.linspace(0.0, 1.0, N_IMAGES)
        path = physical_to_flat(model_coordinates(model, s_values))
        common = load_common_path("inputs/common_initial_path.npz")
        np.testing.assert_allclose(path, common, atol=2e-14, rtol=0.0)

    def test_endpoints_are_fixed_after_training(self):
        model = FixedEndpointPathNet(width=24, depth=2, seed=7)
        s_values = np.linspace(0.0, 1.0, 20)[1:-1]
        straight = REACTANT + s_values[:, None] * (PRODUCT - REACTANT)
        targets = straight.copy()
        targets[:, 0] -= 0.15 * np.sin(np.pi * s_values)
        targets[:, 1] -= 0.10 * np.sin(np.pi * s_values)
        fit_path_targets(
            model,
            s_values,
            targets,
            smoothness_weight=0.0,
            adam_steps=120,
            lbfgs_steps=80,
        )
        endpoints = model_coordinates(model, [0.0, 1.0])
        np.testing.assert_array_equal(endpoints[0], REACTANT)
        np.testing.assert_array_equal(endpoints[1], PRODUCT)

    def test_parameter_checkpoint_is_immutable_and_restores_exactly(self):
        model = FixedEndpointPathNet(width=20, depth=2, seed=8)
        checkpoint = parameter_vector(model)
        checkpoint_copy = checkpoint.copy()
        load_parameter_vector(model, checkpoint)

        s_values = np.linspace(0.0, 1.0, 18)[1:-1]
        straight = REACTANT + s_values[:, None] * (PRODUCT - REACTANT)
        targets = straight - 0.12 * np.sin(np.pi * s_values)[:, None]
        fit_path_targets(
            model,
            s_values,
            targets,
            smoothness_weight=0.0,
            adam_steps=20,
            lbfgs_steps=10,
        )

        # Training after loading must not mutate the NumPy rollback checkpoint.
        np.testing.assert_array_equal(checkpoint, checkpoint_copy)
        self.assertGreater(
            np.linalg.norm(parameter_vector(model) - checkpoint),
            1e-8,
        )
        load_parameter_vector(model, checkpoint)
        np.testing.assert_array_equal(parameter_vector(model), checkpoint)

    def test_autograd_tangent_of_initial_path_is_exact(self):
        model = FixedEndpointPathNet(width=12, depth=2, seed=9)
        _, tangent, speed = normalized_autograd_tangent(
            model,
            np.linspace(0.0, 1.0, 31),
        )
        expected = (PRODUCT - REACTANT) / np.linalg.norm(PRODUCT - REACTANT)
        np.testing.assert_allclose(
            tangent,
            np.broadcast_to(expected, tangent.shape),
            atol=2e-14,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            speed,
            np.linalg.norm(PRODUCT - REACTANT),
            atol=2e-14,
            rtol=0.0,
        )

    def test_equal_arclength_sampling_has_small_spring_residual(self):
        model = FixedEndpointPathNet(width=16, depth=2, seed=11)
        _, path = equal_arclength_sample(model, image_count=N_IMAGES)
        lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self.assertLess(np.max(np.abs(np.diff(lengths))), 2e-13)

    def test_stratified_jitter_is_reproducible_ordered_and_small(self):
        count = 20
        jitter = 0.30
        first = stratified_jitter_fractions(
            count,
            17,
            jitter=jitter,
            seed=314,
        )
        second = stratified_jitter_fractions(
            count,
            17,
            jitter=jitter,
            seed=314,
        )
        base = np.arange(1, count + 1) / float(count + 1)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(np.diff(first) > 0.0))
        self.assertLessEqual(
            np.max(np.abs(first - base)),
            jitter / float(count + 1),
        )
        self.assertFalse(
            np.array_equal(
                first,
                stratified_jitter_fractions(
                    count,
                    18,
                    jitter=jitter,
                    seed=314,
                ),
            )
        )

    def test_requested_arclength_fractions_are_inverted_exactly_for_line(self):
        model = FixedEndpointPathNet(width=16, depth=2, seed=12)
        fractions = stratified_jitter_fractions(
            20,
            4,
            jitter=0.3,
            seed=99,
        )
        parameter_s = arclength_parameters_for_fractions(
            model,
            fractions,
            dense_count=1025,
        )
        np.testing.assert_allclose(parameter_s, fractions, atol=2e-14, rtol=0)

    def test_polyline_arclength_gauge_preserves_endpoints_and_spacing(self):
        path = np.array(
            [[0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [1.0, 1.0]]
        )
        resampled = resample_polyline_equal_arclength(path, count=22)
        np.testing.assert_array_equal(resampled[0], path[0])
        np.testing.assert_array_equal(resampled[-1], path[-1])
        chord_lengths = np.linalg.norm(np.diff(resampled, axis=0), axis=1)
        # One sample interval straddles the corner, so only compare the
        # straight portions against the intended arclength increment.
        expected = 2.0 / 21.0
        away_from_corner = chord_lengths > 0.9 * expected
        np.testing.assert_allclose(
            chord_lengths[away_from_corner],
            expected,
            atol=2e-15,
            rtol=0.0,
        )

    def test_dense_initial_metrics_agree_between_representations(self):
        model = FixedEndpointPathNet(width=16, depth=2, seed=13)
        common = load_common_path("inputs/common_initial_path.npz")
        neural = dense_neural_metrics(model, dense_count=2049)
        linear = dense_linear_metrics(common, subdivisions=100)
        self.assertAlmostEqual(
            neural["dense_barrier"],
            linear["dense_barrier"],
            places=5,
        )
        self.assertAlmostEqual(
            neural["dense_i_perp"] / linear["dense_i_perp"],
            1.0,
            places=4,
        )

    def test_torch_leps_matches_numpy_and_supports_second_backprop(self):
        points_array = np.array(
            [[0.9, 4.3], [1.2, 1.0], [3.1, 2.7]],
            dtype=np.float64,
        )
        points = torch.tensor(
            points_array,
            dtype=torch.float64,
            requires_grad=True,
        )
        energy, force = torch_leps_energy_force(
            points,
            create_graph=True,
        )
        expected_energy, expected_gradient = leps_energy_gradient(
            points_array[:, 0],
            points_array[:, 1],
        )
        np.testing.assert_allclose(
            energy.detach().numpy(),
            expected_energy,
            atol=2e-14,
            rtol=2e-14,
        )
        np.testing.assert_allclose(
            force.detach().numpy(),
            -expected_gradient,
            atol=2e-13,
            rtol=2e-13,
        )
        second_backprop = torch.autograd.grad(
            torch.sum(force**2),
            points,
        )[0]
        self.assertTrue(torch.all(torch.isfinite(second_backprop)))

    def test_adaptive_nn_ivm_joint_oracle_is_cached_and_differentiable(self):
        fractions = np.arange(1, 21, dtype=np.float64) / 21.0
        points_array = (
            REACTANT[None, :]
            + fractions[:, None] * (PRODUCT - REACTANT)[None, :]
        )
        backend = LEPSBackend(label="test-adaptive-nn-ivm-oracle")
        evaluators = make_evaluators(backend, count=22)

        points = torch.tensor(
            points_array,
            dtype=torch.float64,
            requires_grad=True,
        )
        energies, forces = audited_joint_energy_force(
            points,
            evaluators[1:-1],
        )
        energies.sum().backward()
        expected_energy, expected_gradient = leps_energy_gradient(
            points_array[:, 0],
            points_array[:, 1],
        )
        np.testing.assert_allclose(
            energies.detach().numpy(),
            expected_energy,
            atol=2e-14,
            rtol=2e-14,
        )
        np.testing.assert_allclose(
            forces.detach().numpy(),
            -expected_gradient,
            atol=2e-13,
            rtol=2e-13,
        )
        np.testing.assert_allclose(
            points.grad.detach().numpy(),
            expected_gradient,
            atol=2e-13,
            rtol=2e-13,
        )
        self.assertEqual(backend.raw_joint_evaluations, 20)

        repeated = torch.tensor(
            points_array,
            dtype=torch.float64,
            requires_grad=True,
        )
        repeated_energy, _ = audited_joint_energy_force(
            repeated,
            evaluators[1:-1],
        )
        repeated_energy.sum().backward()
        self.assertEqual(backend.raw_joint_evaluations, 20)
        counters = counter_arrays(evaluators)
        np.testing.assert_array_equal(
            counters["joint_evaluations"][1:-1],
            np.ones(20, dtype=int),
        )
        np.testing.assert_array_equal(
            counters["cache_hits"][1:-1],
            np.ones(20, dtype=int),
        )

    def test_continuous_production_runs_share_path_and_close_counter_audit(self):
        files = (
            ROOT
            / "runs"
            / "continuous_v3"
            / "nn_ivm_adaptive_al_a100_j015_b500.npz",
            ROOT
            / "runs"
            / "continuous_v2"
            / "gmam_autodiff20_production_lr001_b500.npz",
        )
        initial_paths = []
        for path in files:
            with np.load(path, allow_pickle=False) as run:
                initial_paths.append(run["initial_path"].copy())
                self.assertEqual(int(run["collocation_count"]), 20)
                self.assertTrue(bool(run["counter_audit_consistent"]))
                self.assertEqual(
                    int(np.sum(run["counter_vector"][-1])),
                    int(run["backend_raw_joint_evaluations"]),
                )
                for key in (
                    "energy_request_vector",
                    "force_request_vector",
                    "cache_hit_vector",
                    "counter_vector",
                ):
                    self.assertEqual(
                        run[key].shape[1],
                        int(run["collocation_count"]) + 2,
                    )
        np.testing.assert_array_equal(initial_paths[0], initial_paths[1])

        with np.load(files[0], allow_pickle=False) as nn_ivm:
            accepted = np.asarray(nn_ivm["accepted"], dtype=bool)
            accepted[0] = False
            self.assertTrue(
                np.all(nn_ivm["constraint_max"][accepted] <= 5e-3)
            )
            model = FixedEndpointPathNet(
                width=int(nn_ivm["width"]),
                depth=int(nn_ivm["depth"]),
                fourier_order=int(nn_ivm["fourier_order"]),
                seed=int(nn_ivm["seed"]),
            )
            load_parameter_vector(model, nn_ivm["model_state"][-1])
            endpoints = model_coordinates(model, [0.0, 1.0])
            np.testing.assert_array_equal(endpoints[0], REACTANT)
            np.testing.assert_array_equal(endpoints[1], PRODUCT)

        with np.load(files[1], allow_pickle=False) as gmam:
            self.assertIn(
                "no explicit or FD Hessian",
                str(gmam["hessian_definition"]),
            )


if __name__ == "__main__":
    unittest.main()
