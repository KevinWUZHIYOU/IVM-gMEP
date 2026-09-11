from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

import frozen_discrete_b040 as discrete
import run_ivm_v9_discrete_exact_candidate_budget_solver as candidate_budget
import run_ivm_v9_discrete_ffrbk_takeover_exact_noreparam_shim as exact
import run_stage4_ala_discrete_exact_noreparam_alpha_grid_member as member


class FakeEvaluationBudgetReached(RuntimeError):
    pass


class MockEvaluator:
    def __init__(self, image_index, budget):
        self.image_index = int(image_index)
        self.budget = budget
        self.cache = {}
        self.counters = SimpleNamespace(
            joint_evaluations=1,
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

    def seed(self, position):
        position = np.asarray(position, dtype=np.float64)
        force = np.zeros_like(position)
        force[1] = -1.0
        self.cache[self._key(position)] = (float(position[1]), force)

    def has_cached(self, position):
        return self._key(position) in self.cache

    def query(self, position, property_name="both"):
        position = np.asarray(position, dtype=np.float64)
        key = self._key(position)
        if key in self.cache:
            self.counters.cache_hits += 1
            energy, force = self.cache[key]
        else:
            force = np.zeros_like(position)
            force[1] = -1.0
            energy = float(position[1])
            self.cache[key] = (energy, force.copy())
            self.counters.joint_evaluations += 1
            if 0 < self.image_index <= 20:
                self.budget.successful_internal_joint_evaluations += 1
        if property_name == "energy":
            return energy
        if property_name == "force":
            return force.copy()
        return energy, force.copy()


class IdentityBank:
    def solve_normal(self, image_index, vector, tangent, diagonal_shift=0.0):
        del image_index, tangent, diagonal_shift
        return np.asarray(vector, dtype=float)


def candidate_problem(total_budget=5000, cache_first_trial_for=0):
    path = np.zeros((22, 3), dtype=float)
    path[:, 0] = np.linspace(-1.0, 1.0, 22)
    forces = np.zeros_like(path)
    forces[:, 1] = -1.0
    energies = path[:, 1].copy()
    tangents = np.zeros((20, 3), dtype=float)
    tangents[:, 0] = 1.0
    budget = SimpleNamespace(
        budget=int(total_budget),
        successful_internal_joint_evaluations=20,
        full_band_preflight_calls=0,
        full_band_preflight_crops=0,
        full_band_preflight_zero_step_stops=0,
        complete_band_budget_stop_count=0,
        ala_core_module=SimpleNamespace(
            EvaluationBudgetReached=FakeEvaluationBudgetReached
        ),
    )
    evaluators = [MockEvaluator(index, budget) for index in range(22)]
    for evaluator, position in zip(evaluators, path):
        evaluator.seed(position)
    # Identity H0 and unit normal force produce a first y-step of -0.1.
    trial = path.copy()
    trial[1:-1, 1] = -0.1
    for index in range(cache_first_trial_for):
        evaluators[index + 1].seed(trial[index + 1])
    return path, energies, forces, tangents, evaluators, budget


class ExactNoReparamRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    @staticmethod
    def _replace(command, flag, value):
        changed = list(command)
        index = changed.index(flag)
        changed[index + 1] = value
        return changed

    def _assert_solver_results_equal(self, left, right, path="result"):
        """Bitwise differential oracle, except for the composed policy ID."""

        if isinstance(left, dict):
            self.assertIsInstance(right, dict, path)
            self.assertEqual(set(left), set(right), path)
            for key in left:
                if key == "implementation_id":
                    continue
                self._assert_solver_results_equal(
                    left[key], right[key], f"{path}.{key}"
                )
            return
        if isinstance(left, (list, tuple)):
            self.assertIsInstance(right, type(left), path)
            self.assertEqual(len(left), len(right), path)
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                self._assert_solver_results_equal(
                    left_item, right_item, f"{path}[{index}]"
                )
            return
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=path)
            return
        self.assertEqual(left, right, path)

    def test_registered_member_command_passes_exact_runtime_validator(self):
        for alpha in member.ALPHAS:
            command, _ = member.single_run_command(self.root, alpha)
            controls, mode, eta, runner_argv = (
                exact.grid_runtime.extract_runtime_arguments(command)
            )
            self.assertEqual(controls.inner_max_steps, 5000)
            with exact.grid_runtime.configured_selector_policy(
                exact.fixed_selector, controls
            ) as selector:
                exact.validate_exact_no_reparam_runner_argv(
                    selector, runner_argv, controls, mode, eta
                )
                identifier = selector.selector_implementation_id("base", mode)
                self.assertIn("c5000", identifier)

    def test_runtime_rejects_reparameterization_or_small_cap(self):
        command, _ = member.single_run_command(self.root, 1.0)
        for flag, value in (
            ("--reparam-every", "1"),
            ("--discrete-inner-max-steps", "250"),
            ("--discrete-residual-ratio", "0.8"),
        ):
            changed = self._replace(command, flag, value)
            with self.assertRaises(ValueError):
                controls, mode, eta, runner_argv = (
                    exact.grid_runtime.extract_runtime_arguments(changed)
                )
                with exact.grid_runtime.configured_selector_policy(
                    exact.fixed_selector, controls
                ) as selector:
                    exact.validate_exact_no_reparam_runner_argv(
                        selector, runner_argv, controls, mode, eta
                    )

    def test_candidate_preflight_accepts_one_hit_plus_nineteen_misses(self):
        problem = candidate_problem(cache_first_trial_for=1)
        path, energies, forces, tangents, evaluators, budget = problem
        original = discrete.solve_frozen_discrete_b040_band
        installed = candidate_budget.install_candidate_aware_budget_solver(
            discrete, budget
        )
        try:
            result = discrete.solve_frozen_discrete_b040_band(
                path,
                energies,
                forces,
                tangents,
                evaluators,
                1.0,
                0.1,
                5000,
                inner_max_steps=1,
                stop_mode="c10",
                relative_residual=0.1,
                history_size=7,
                kkt_atol=0.0,
                preconditioner_bank=IdentityBank(),
            )
            self.assertEqual(result[8]["joint_steps"], 1)
            self.assertEqual(budget.successful_internal_joint_evaluations, 39)
            self.assertEqual(
                installed.events[0].required_internal_misses, 19
            )
            counters = result[7][0]["joint_evaluations"][1:-1]
            self.assertEqual(int(np.sum(counters)), 39)
            self.assertEqual(int(counters[0]), 1)
            self.assertTrue(np.all(counters[1:] == 2))
        finally:
            installed.restore()
        self.assertIs(discrete.solve_frozen_discrete_b040_band, original)

    def test_candidate_solver_matches_frozen_base_when_budget_nonbinding(self):
        kwargs = {
            "alpha": 1.0,
            "max_step": 0.1,
            "max_new_calls": 5000,
            "inner_max_steps": 3,
            "stop_mode": "c10",
            "relative_residual": 0.1,
            "history_size": 7,
            "kkt_atol": 0.0,
            "preconditioner_bank": IdentityBank(),
        }
        for first_trial_hits in (0, 1):
            with self.subTest(first_trial_hits=first_trial_hits):
                base_problem = candidate_problem(
                    cache_first_trial_for=first_trial_hits
                )
                base_result = discrete.solve_frozen_discrete_b040_band(
                    *base_problem[:5], **kwargs
                )

                candidate_problem_data = candidate_problem(
                    cache_first_trial_for=first_trial_hits
                )
                installed = candidate_budget.install_candidate_aware_budget_solver(
                    discrete, candidate_problem_data[5]
                )
                try:
                    candidate_result = discrete.solve_frozen_discrete_b040_band(
                        *candidate_problem_data[:5], **kwargs
                    )
                finally:
                    installed.restore()
                self._assert_solver_results_equal(base_result, candidate_result)

    def test_candidate_refusal_raises_after_charging_prior_trials(self):
        problem = candidate_problem(total_budget=61)
        path, energies, forces, tangents, evaluators, budget = problem
        installed = candidate_budget.install_candidate_aware_budget_solver(
            discrete, budget
        )
        try:
            with self.assertRaises(FakeEvaluationBudgetReached):
                discrete.solve_frozen_discrete_b040_band(
                    path,
                    energies,
                    forces,
                    tangents,
                    evaluators,
                    1.0,
                    0.1,
                    5000,
                    inner_max_steps=5000,
                    stop_mode="c10",
                    relative_residual=0.1,
                    history_size=7,
                    kkt_atol=0.0,
                    preconditioner_bank=IdentityBank(),
                )
            self.assertEqual(budget.successful_internal_joint_evaluations, 60)
            self.assertEqual([e.stopped_before_query for e in installed.events],
                             [False, False, True])
            self.assertEqual(budget.complete_band_budget_stop_count, 1)
        finally:
            installed.restore()

    def test_completion_audit_accepts_absolute_zero_as_normalized_converged(self):
        path = np.zeros((22, 3), dtype=float)
        endpoint = path.copy()
        endpoint[1:-1, 1] = -0.25
        counter_vector = np.ones(22, dtype=int)
        counter_vector[1:-1] = 2
        synchronized_counters = [{"joint_evaluations": counter_vector}]

        def solve(
            path,
            energies,
            forces,
            tangents,
            evaluators,
            alpha,
            max_step,
            max_new_calls,
            **kwargs,
        ):
            del alpha, max_step, max_new_calls, kwargs
            report = {
                "stop_reason": "evaluated-absolute-natural-residual",
                "joint_steps": 1,
                "global_relative_residual": 0.0,
            }
            return (
                endpoint,
                energies,
                forces,
                np.ones(20, dtype=int),
                [endpoint.copy()],
                [energies.copy()],
                [forces.copy()],
                synchronized_counters,
                report,
            )

        frozen = SimpleNamespace(solve_frozen_discrete_b040_band=solve)
        evaluators = [
            SimpleNamespace(counters=SimpleNamespace(joint_evaluations=1))
            for _ in range(22)
        ]
        installed = exact.install_exact_solve_completion_audit(frozen)
        try:
            frozen.solve_frozen_discrete_b040_band(
                path,
                np.zeros(22),
                np.zeros_like(path),
                np.zeros((20, 3)),
                evaluators,
                1.0,
                1.0,
                5000,
            )
        finally:
            installed.restore()
        self.assertEqual(len(installed.events), 1)
        event = installed.events[0]
        self.assertEqual(
            event.stop_reason, "evaluated-absolute-natural-residual"
        )
        self.assertEqual(event.exact_endpoint_nf, 40)
        self.assertEqual(event.global_relative_residual, 0.0)
        np.testing.assert_array_equal(event.raw_exact_endpoint_path, endpoint)

    def test_runner_alias_and_implementation_identity_restore(self):
        original_solve = object()
        installation = SimpleNamespace(
            original_solve=original_solve,
            original_implementation_id="frozen-base",
        )
        runner = SimpleNamespace(
            solve_frozen_discrete_b040_band=object(),
            DISCRETE_B040_IMPLEMENTATION_ID="composed",
        )
        exact.restore_runner_candidate_identity(runner, installation)
        self.assertIs(runner.solve_frozen_discrete_b040_band, original_solve)
        self.assertEqual(runner.DISCRETE_B040_IMPLEMENTATION_ID, "frozen-base")


if __name__ == "__main__":
    unittest.main()
