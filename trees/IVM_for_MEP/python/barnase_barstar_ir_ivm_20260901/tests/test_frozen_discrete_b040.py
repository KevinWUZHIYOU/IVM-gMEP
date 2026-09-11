from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

import frozen_discrete_b040 as discrete


class MockEvaluator:
    def __init__(self, potential):
        self.potential = potential
        self.cache = {}
        self.counters = SimpleNamespace(
            joint_evaluations=0,
            energy_requests=0,
            force_requests=0,
            cache_hits=0,
            total_query_events=0,
            both_requests=0,
            budget_blocked_misses=0,
        )

    @staticmethod
    def _key(position):
        return np.ascontiguousarray(
            np.asarray(position, dtype=np.float64)
        ).tobytes()

    def has_cached(self, position):
        return self._key(position) in self.cache

    def query(self, position, property_name="both"):
        self.counters.total_query_events += 1
        if property_name in {"energy", "both"}:
            self.counters.energy_requests += 1
        if property_name in {"force", "both"}:
            self.counters.force_requests += 1
        if property_name == "both":
            self.counters.both_requests += 1
        key = self._key(position)
        if key in self.cache:
            self.counters.cache_hits += 1
            energy, force = self.cache[key]
        else:
            energy, force = self.potential(
                np.asarray(position, dtype=np.float64)
            )
            energy = float(energy)
            force = np.asarray(force, dtype=np.float64)
            self.cache[key] = (energy, force.copy())
            self.counters.joint_evaluations += 1
        if property_name == "energy":
            return energy
        if property_name == "force":
            return force.copy()
        return energy, force.copy()


class IdentityH0:
    def __init__(self, scale=1.0):
        self.scale = float(scale)
        self.calls = []

    def solve_normal(
        self, image_index, vector, tangent, diagonal_shift=0.0
    ):
        self.calls.append(
            (int(image_index), float(diagonal_shift), np.array(vector, copy=True))
        )
        return self.scale * np.asarray(vector, dtype=np.float64)


def one_image_problem(potential, internal=(0.0, 1.0, 0.0)):
    path = np.asarray(
        [(-1.0, 0.0, 0.0), internal, (1.0, 0.0, 0.0)],
        dtype=np.float64,
    )
    evaluators = [MockEvaluator(potential) for _ in range(3)]
    energies, forces = discrete.evaluate_path(path, evaluators)
    tangents = np.asarray([(1.0, 0.0, 0.0)], dtype=np.float64)
    return path, energies, forces, tangents, evaluators


def quadratic(stiffness):
    stiffness = float(stiffness)

    def potential(position):
        energy = 0.5 * stiffness * float(position[1] ** 2)
        force = np.zeros_like(position)
        force[1] = -stiffness * position[1]
        return energy, force

    return potential


def linear(position):
    force = np.zeros_like(position)
    force[1] = -1.0
    return float(position[1]), force


class DiscreteDirectionTests(unittest.TestCase):
    def test_h0_shift_and_normal_projection(self):
        bank = IdentityH0()
        covector = np.asarray([(3.0, 4.0, 0.0)])
        tangents = np.asarray([(1.0, 0.0, 0.0)])
        result = discrete.apply_block_h0_inverse(
            covector, tangents, bank, alpha=2.0
        )
        self.assertEqual(bank.calls[0][1], 0.5)
        np.testing.assert_array_equal(result, [[0.0, 4.0, 0.0]])

    def test_history_free_tie_selects_h0(self):
        gradient = np.asarray([(0.0, -1.0, 0.0)])
        tangents = np.asarray([(1.0, 0.0, 0.0)])
        raw = np.asarray([(0.0, 2.0, 0.0)])
        h0 = discrete.screen_direction(
            "H0", raw, gradient, tangents, max_step=0.1
        )
        lbfgs = discrete.screen_direction(
            "LBFGS", raw.copy(), gradient, tangents, max_step=0.1
        )
        self.assertEqual(h0["capped_decrease"], lbfgs["capped_decrease"])
        self.assertIs(discrete.select_direction(h0, lbfgs), h0)

    def test_selection_uses_capped_decrease(self):
        gradient = np.asarray([(0.0, -1.0, 0.0, 0.0, -1.0, 0.0)])
        tangents = np.asarray([(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)])
        raw_h0 = np.asarray([(0.0, 100.0, 0.0, 0.0, 0.0, 0.0)])
        raw_lbfgs = np.asarray([(0.0, 2.0, 0.0, 0.0, 2.0, 0.0)])
        h0 = discrete.screen_direction(
            "H0", raw_h0, gradient, tangents, max_step=0.1
        )
        lbfgs = discrete.screen_direction(
            "LBFGS", raw_lbfgs, gradient, tangents, max_step=0.1
        )
        self.assertGreater(h0["raw_decrease"], lbfgs["raw_decrease"])
        self.assertAlmostEqual(h0["capped_decrease"], 0.1)
        self.assertAlmostEqual(lbfgs["capped_decrease"], 0.2)
        self.assertEqual(discrete.select_direction(h0, lbfgs)["label"], "LBFGS")

    def test_invalid_fallback_and_double_invalid(self):
        gradient = np.asarray([(0.0, 1.0, 0.0)])
        tangents = np.asarray([(1.0, 0.0, 0.0)])
        invalid = discrete.screen_direction(
            "H0", [(0.0, 1.0, 0.0)], gradient, tangents, 1.0
        )
        valid = discrete.screen_direction(
            "LBFGS", [(0.0, -1.0, 0.0)], gradient, tangents, 1.0
        )
        self.assertIs(discrete.select_direction(invalid, valid), valid)
        with self.assertRaisesRegex(
            discrete.DiscreteB040SafetyError, "both H0 and LBFGS"
        ):
            discrete.select_direction(invalid, dict(invalid))

    def test_joint_cap_uses_one_scale_and_stays_normal(self):
        raw = np.asarray(
            [(3.0, 2.0, 0.0), (0.0, 4.0, 0.0)], dtype=np.float64
        )
        tangents = np.asarray(
            [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)], dtype=np.float64
        )
        step, scale = discrete.cap_joint_direction(raw, tangents, 0.1)
        self.assertAlmostEqual(scale, 0.025)
        np.testing.assert_allclose(step, discrete.project_normal(raw, tangents) * scale)
        np.testing.assert_allclose(np.sum(step * tangents, axis=1), 0.0)
        self.assertLessEqual(
            np.max(np.linalg.norm(step.reshape(-1, 3), axis=1)), 0.1
        )

    def test_curvature_gate_fifo_copy_and_history_corruption(self):
        history_s, history_y = [], []
        for index in range(9):
            s_value = np.asarray([[0.0, float(index + 1), 0.0]])
            y_value = s_value.copy()
            status = discrete.append_curvature(
                history_s, history_y, s_value, y_value, 7
            )
            self.assertTrue(status.startswith("accepted"))
            s_value[:] = 999.0
        self.assertEqual(len(history_s), 7)
        self.assertNotEqual(float(history_s[-1][0, 1]), 999.0)
        rejected = discrete.append_curvature(
            history_s,
            history_y,
            np.asarray([[0.0, 1.0, 0.0]]),
            np.asarray([[0.0, -1.0, 0.0]]),
            7,
        )
        self.assertTrue(rejected.startswith("rejected"))
        with self.assertRaisesRegex(
            discrete.DiscreteB040SafetyError, "history lengths"
        ):
            discrete.joint_lbfgs_direction(
                np.ones((1, 3)), [np.ones((1, 3))], [], lambda value: value
            )
        with self.assertRaisesRegex(
            discrete.DiscreteB040SafetyError, "pair shape"
        ):
            discrete.joint_lbfgs_direction(
                np.ones((1, 3)),
                [np.ones((2, 3))],
                [np.ones((2, 3))],
                lambda value: value,
            )


class DiscreteSolverTests(unittest.TestCase):
    def run_solver(
        self,
        potential,
        *,
        stop_mode,
        inner_max_steps=10,
        max_step=10.0,
        alpha=np.inf,
        internal=(0.0, 1.0, 0.0),
    ):
        path, energies, forces, tangents, evaluators = one_image_problem(
            potential, internal=internal
        )
        original_path = path.copy()
        original_tangents = tangents.copy()
        result = discrete.solve_frozen_discrete_b040_band(
            path,
            energies,
            forces,
            tangents,
            evaluators,
            alpha,
            max_step,
            inner_max_steps,
            inner_max_steps=inner_max_steps,
            stop_mode=stop_mode,
            relative_residual=0.8,
            history_size=7,
            preconditioner_bank=IdentityH0(),
        )
        np.testing.assert_array_equal(path, original_path)
        np.testing.assert_array_equal(tangents, original_tangents)
        return result, evaluators, original_path, original_tangents

    def test_r08_uses_norm_ratio_and_switches_to_lbfgs(self):
        result, _, _, _ = self.run_solver(quadratic(0.15), stop_mode="r08")
        used = result[3]
        report = result[8]
        self.assertEqual(report["joint_steps"], 2)
        self.assertEqual(report["direction_selection"].tolist(), ["H0", "LBFGS"])
        self.assertEqual(report["stop_reason"], "evaluated-absolute-natural-residual")
        np.testing.assert_array_equal(used, [2])
        self.assertAlmostEqual(report["global_relative_residual"], 0.0, places=12)

    def test_r08c10_is_exact_alias_for_archived_r08(self):
        archived, _, _, _ = self.run_solver(
            quadratic(0.25), stop_mode="r08"
        )
        canonical, _, _, _ = self.run_solver(
            quadratic(0.25), stop_mode="r08c10"
        )
        for archived_value, canonical_value in zip(
            archived[:8], canonical[:8]
        ):
            if isinstance(archived_value, list):
                self.assertEqual(len(archived_value), len(canonical_value))
                for archived_item, canonical_item in zip(
                    archived_value, canonical_value
                ):
                    if isinstance(archived_item, dict):
                        self.assertEqual(
                            archived_item.keys(), canonical_item.keys()
                        )
                        for key in archived_item:
                            np.testing.assert_array_equal(
                                archived_item[key], canonical_item[key]
                            )
                    else:
                        np.testing.assert_array_equal(
                            archived_item, canonical_item
                        )
            else:
                np.testing.assert_array_equal(
                    archived_value, canonical_value
                )
        archived_report = archived[8]
        canonical_report = canonical[8]
        self.assertEqual(archived_report.keys(), canonical_report.keys())
        for key in archived_report:
            np.testing.assert_array_equal(
                archived_report[key], canonical_report[key]
            )
        self.assertEqual(canonical_report["joint_steps"], 1)
        self.assertEqual(
            canonical_report["stop_reason"],
            "evaluated-relative-natural-residual",
        )

    def test_c10_means_ten_evaluated_steps_not_atomic_cap(self):
        result, _, _, _ = self.run_solver(
            linear,
            stop_mode="c10",
            inner_max_steps=10,
            max_step=0.01,
            internal=(0.0, 0.0, 0.0),
        )
        final_path, used, traces, report = result[0], result[3], result[4], result[8]
        self.assertEqual(report["joint_steps"], 10)
        self.assertEqual(report["stop_reason"], "evaluated-inner-max-steps")
        np.testing.assert_array_equal(used, [10])
        self.assertAlmostEqual(final_path[1, 1], -0.1)
        self.assertEqual(len(traces), 10)
        self.assertTrue(np.all(report["direction_selection"] == "H0"))

    def test_c10_exact_stationary_stops_without_extra_query(self):
        result, _, _, _ = self.run_solver(quadratic(1.0), stop_mode="c10")
        report = result[8]
        self.assertEqual(report["joint_steps"], 1)
        self.assertEqual(
            report["stop_reason"], "evaluated-absolute-natural-residual"
        )
        np.testing.assert_array_equal(result[3], [1])

    def test_returned_states_are_evaluated_and_endpoints_fixed(self):
        result, evaluators, reference_path, tangents = self.run_solver(
            quadratic(0.15), stop_mode="r08"
        )
        final_path, final_energies, final_forces = result[:3]
        traces, trace_energies, trace_forces = result[4:7]
        report = result[8]
        self.assertTrue(report["all_returned_states_evaluated"])
        self.assertTrue(report["endpoints_bitwise_fixed"])
        np.testing.assert_array_equal(final_path[[0, -1]], reference_path[[0, -1]])
        self.assertEqual(len(traces), report["joint_steps"])
        np.testing.assert_array_equal(traces[-1], final_path)
        np.testing.assert_array_equal(trace_energies[-1], final_energies)
        np.testing.assert_array_equal(trace_forces[-1], final_forces)
        self.assertTrue(evaluators[1].has_cached(final_path[1]))
        displacement = final_path[1:-1] - reference_path[1:-1]
        np.testing.assert_allclose(np.sum(displacement * tangents, axis=1), 0.0)
        self.assertEqual(
            report["frozen_path_sha256"],
            hashlib.sha256(np.ascontiguousarray(reference_path).tobytes()).hexdigest(),
        )

    def test_input_validation_and_no_hidden_reparameterization(self):
        path, energies, forces, tangents, evaluators = one_image_problem(
            quadratic(1.0)
        )
        kwargs = dict(
            path=path,
            energies=energies,
            forces=forces,
            tangents=tangents,
            evaluators=evaluators,
            alpha=1.0,
            max_step=1.0,
            max_new_calls=1,
            preconditioner_bank=IdentityH0(),
        )
        bad_tangent = tangents * 2.0
        with self.assertRaisesRegex(ValueError, "unit vectors"):
            discrete.solve_frozen_discrete_b040_band(
                **{**kwargs, "tangents": bad_tangent}
            )
        with self.assertRaisesRegex(ValueError, "every internal image"):
            discrete.solve_frozen_discrete_b040_band(
                **{**kwargs, "active_images": np.ones(2, dtype=bool)}
            )
        with self.assertRaisesRegex(ValueError, r"positive or \+inf"):
            discrete.solve_frozen_discrete_b040_band(
                **{**kwargs, "alpha": 0.0}
            )
        source = Path(discrete.__file__).read_text()
        self.assertNotIn("central_spring_force_equilibrium", source)


if __name__ == "__main__":
    unittest.main()
