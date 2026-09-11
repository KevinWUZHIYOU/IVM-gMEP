import numpy as np
import torch
import unittest
from ase import Atoms
from ase.mep.neb import NEB, NEBState

from leps_core import (
    A_LEPS,
    B_LEPS,
    C_LEPS,
    D_AB,
    D_AC,
    D_BC,
    MORSE_ALPHA,
    R0,
    AuditedLEPSCalculator,
    ImageEvaluator,
    LEPSBackend,
    array_sha256,
    audit_counter_consistency,
    build_common_idpp_path,
    equidistance_diagnostics,
    improved_hermite_equal_chord,
    improved_tangents,
    leps_energy_gradient,
    make_evaluators,
    physical_to_flat,
    reference_saddle,
)


def historical_torch_energy(point):
    point = torch.tensor(point, dtype=torch.float64, requires_grad=True)

    def q(r, d):
        dr = r - R0
        return d / 2.0 * (
            1.5 * torch.exp(-2.0 * MORSE_ALPHA * dr)
            - torch.exp(-MORSE_ALPHA * dr)
        )

    def j(r, d):
        dr = r - R0
        return d / 4.0 * (
            torch.exp(-2.0 * MORSE_ALPHA * dr)
            - 6.0 * torch.exp(-MORSE_ALPHA * dr)
        )

    r_ab, r_bc = point
    r_ac = r_ab + r_bc
    q_ab, q_bc, q_ac = q(r_ab, D_AB), q(r_bc, D_BC), q(r_ac, D_AC)
    j_ab, j_bc, j_ac = j(r_ab, D_AB), j(r_bc, D_BC), j(r_ac, D_AC)
    pa, pb, pc = 1 + A_LEPS, 1 + B_LEPS, 1 + C_LEPS
    radicand = (
        j_ab**2 / pa**2
        + j_bc**2 / pb**2
        + j_ac**2 / pc**2
        - j_ab * j_bc / (pa * pb)
        - j_bc * j_ac / (pb * pc)
        - j_ab * j_ac / (pa * pc)
    )
    energy = q_ab / pa + q_bc / pb + q_ac / pc - torch.sqrt(
        torch.clamp(radicand, min=0.0)
    )
    gradient = torch.autograd.grad(energy, point)[0]
    return energy.item(), gradient.detach().numpy()


class TestLEPSCore(unittest.TestCase):
    def test_energy_and_force_match_historical_notebook(self):
        points = [
            [0.74200064, 8.0],
            [8.0, 0.74200064],
            [1.14937799, 0.86246877],
            [2.0, 1.1],
            [3.0, 4.0],
        ]
        for point in points:
            with self.subTest(point=point):
                expected_energy, expected_gradient = historical_torch_energy(point)
                energy, gradient = leps_energy_gradient(point[0], point[1])
                self.assertAlmostEqual(float(energy), expected_energy, places=12)
                np.testing.assert_allclose(
                    gradient, expected_gradient, rtol=2e-13, atol=2e-13
                )

    def test_gradient_matches_finite_difference(self):
        point = np.array([1.7, 1.2])
        _, gradient = leps_energy_gradient(*point)
        step = 1e-6
        finite = np.empty(2)
        for j in range(2):
            offset = np.zeros(2)
            offset[j] = step
            e_plus = leps_energy_gradient(*(point + offset))[0]
            e_minus = leps_energy_gradient(*(point - offset))[0]
            finite[j] = (e_plus - e_minus) / (2.0 * step)
        np.testing.assert_allclose(gradient, finite, rtol=2e-8, atol=2e-8)

    def test_improved_tangent_is_ase_runtime_method(self):
        physical = np.column_stack(
            [
                np.linspace(0.74200064, 8.0, 22),
                np.linspace(8.0, 0.74200064, 22),
            ]
        )
        path = physical_to_flat(physical)
        energies = np.asarray(
            [leps_energy_gradient(*point)[0] for point in physical]
        )
        ours = improved_tangents(path, energies)
        images = [
            Atoms("H2", positions=np.asarray(flat).reshape(2, 3)) for flat in path
        ]
        neb = NEB(images, k=1.0, method="improvedtangent")
        self.assertEqual(type(neb.neb_method).__name__, "ImprovedTangentMethod")
        state = NEBState(neb, images, energies)
        spring1 = state.spring(0)
        direct = []
        for i in range(1, 21):
            spring2 = state.spring(i)
            direct.append(
                neb.neb_method.get_tangent(
                    state, spring1, spring2, i
                ).reshape(-1)
            )
            spring1 = spring2
        np.testing.assert_array_equal(ours, np.asarray(direct))

    def test_joint_cache_and_historical_revisit_rules(self):
        backend = LEPSBackend()
        evaluator = ImageEvaluator(backend, 1)
        a = physical_to_flat([1.0, 2.0])
        b = physical_to_flat([1.1, 2.0])
        evaluator.query(a, "force")
        self.assertEqual(evaluator.counters.joint_evaluations, 1)
        evaluator.query(a, "energy")
        evaluator.query(a, "both")
        self.assertEqual(evaluator.counters.joint_evaluations, 1)
        evaluator.query(b, "energy")
        self.assertEqual(evaluator.counters.joint_evaluations, 2)
        evaluator.query(b, "force")
        evaluator.query(a, "both")
        self.assertEqual(evaluator.counters.joint_evaluations, 2)
        self.assertEqual(backend.raw_joint_evaluations, 2)
        self.assertEqual(evaluator.counters.cache_hits, 4)

    def test_ase_force_then_energy_is_one_joint_evaluation(self):
        backend = LEPSBackend()
        evaluator = ImageEvaluator(backend, 1)
        atoms = Atoms(
            "H2", positions=physical_to_flat([1.2, 1.5]).reshape(2, 3)
        )
        atoms.calc = AuditedLEPSCalculator(evaluator)
        atoms.get_forces()
        atoms.get_potential_energy()
        atoms.get_forces()
        self.assertEqual(evaluator.counters.force_requests, 2)
        self.assertEqual(evaluator.counters.energy_requests, 1)
        self.assertEqual(evaluator.counters.joint_evaluations, 1)
        self.assertEqual(backend.raw_joint_evaluations, 1)

    def test_coordinate_changes_without_property_request_do_not_count(self):
        backend = LEPSBackend()
        evaluator = ImageEvaluator(backend, 1)
        atoms = Atoms(
            "H2", positions=physical_to_flat([1.0, 2.0]).reshape(2, 3)
        )
        atoms.calc = AuditedLEPSCalculator(evaluator)
        atoms.set_positions(physical_to_flat([1.1, 2.0]).reshape(2, 3))
        atoms.set_positions(physical_to_flat([1.2, 2.0]).reshape(2, 3))
        self.assertEqual(evaluator.counters.joint_evaluations, 0)
        atoms.get_potential_energy()
        self.assertEqual(evaluator.counters.joint_evaluations, 1)
        self.assertEqual(backend.raw_joint_evaluations, 1)

    def test_rejected_trial_remains_counted(self):
        backend = LEPSBackend()
        evaluator = ImageEvaluator(backend, 1)
        accepted = physical_to_flat([1.0, 2.0])
        rejected = physical_to_flat([1.3, 2.0])
        evaluator.query(accepted, "both")
        evaluator.query(rejected, "both")
        evaluator.query(accepted, "both")
        self.assertEqual(evaluator.counters.joint_evaluations, 2)
        self.assertEqual(backend.raw_joint_evaluations, 2)

    def test_checkpoint_restores_cache_and_counters(self):
        first_backend = LEPSBackend()
        first = ImageEvaluator(first_backend, 1)
        a = physical_to_flat([1.0, 2.0])
        b = physical_to_flat([1.2, 2.0])
        first.query(a, "force")
        first.query(b, "energy")
        state = first.snapshot()
        backend_state = first_backend.snapshot()

        restored_backend = LEPSBackend()
        restored_backend.restore(backend_state)
        restored = ImageEvaluator(restored_backend, 1)
        restored.restore(state)
        restored.query(a, "energy")
        self.assertEqual(restored.counters.joint_evaluations, 2)
        self.assertEqual(restored_backend.raw_joint_evaluations, 2)
        restored.query(physical_to_flat([1.4, 2.0]), "both")
        self.assertEqual(restored.counters.joint_evaluations, 3)
        self.assertEqual(restored_backend.raw_joint_evaluations, 3)

    def test_all_image_counter_sum_equals_backend(self):
        backend = LEPSBackend()
        evaluators = make_evaluators(backend)
        path, _ = build_common_idpp_path()
        for evaluator, point in zip(evaluators, path):
            evaluator.query(point, "both")
            evaluator.query(point, "energy")
        audit = audit_counter_consistency(backend, evaluators)
        self.assertTrue(audit["consistent"])
        self.assertEqual(audit["backend_raw_joint_evaluations"], 22)

    def test_common_idpp_path_is_deterministic_and_has_fixed_endpoints(self):
        first, source_first = build_common_idpp_path()
        second, source_second = build_common_idpp_path()
        self.assertEqual(source_first, source_second)
        self.assertEqual(source_first, "ASE IDPP")
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(
            first[0], physical_to_flat([0.74200064, 8.0])
        )
        np.testing.assert_array_equal(
            first[-1], physical_to_flat([8.0, 0.74200064])
        )
        self.assertEqual(array_sha256(first), array_sha256(second))

    def test_hermite_projection_equalizes_chords(self):
        path, _ = build_common_idpp_path()
        physical = path[:, [0, 3]].copy()
        physical[1:-1, 1] += 0.15 * np.sin(np.linspace(0.0, np.pi, 20))
        perturbed = physical_to_flat(physical)
        energies = np.asarray(
            [leps_energy_gradient(*point)[0] for point in physical]
        )
        tangents = improved_tangents(perturbed, energies)
        projected, residual = improved_hermite_equal_chord(perturbed, tangents)
        lengths = np.linalg.norm(np.diff(projected, axis=0), axis=1)
        self.assertLess(residual, 1e-10)
        self.assertLess(np.std(lengths) / np.mean(lengths), 1e-10)
        np.testing.assert_array_equal(projected[0], perturbed[0])
        np.testing.assert_array_equal(projected[-1], perturbed[-1])
        diagnostics = equidistance_diagnostics(projected)
        self.assertLess(diagnostics["equidistance_integral"], 1e-15)
        self.assertEqual(
            diagnostics["i_arc"], diagnostics["equidistance_integral"]
        )
        self.assertLess(diagnostics["spring_force_max_unit"], 1e-12)

    def test_unit_spring_force_is_adjacent_chord_length_difference(self):
        physical = np.array(
            [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]]
        )
        diagnostics = equidistance_diagnostics(physical_to_flat(physical))
        self.assertAlmostEqual(diagnostics["spring_force_max_unit"], 1.0)
        self.assertGreater(diagnostics["i_arc"], 0.0)

    def test_reference_stationary_point_is_index_one_saddle(self):
        saddle = reference_saddle()
        self.assertLess(saddle["gradient_norm"], 1e-10)
        self.assertLess(saddle["hessian_eigenvalues"][0], 0.0)
        self.assertGreater(saddle["hessian_eigenvalues"][1], 0.0)
        self.assertAlmostEqual(saddle["forward_barrier"], 1.3430860763, places=8)


if __name__ == "__main__":
    unittest.main()
