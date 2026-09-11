from __future__ import annotations

import inspect
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import frozen_discrete_b040 as discrete
import run_ivm_v9_discrete_exact_candidate_budget_solver as candidate_budget
import run_ivm_v9_discrete_ffrbk_takeover_exact_noreparam_shim as legacy
import run_ivm_v9_discrete_ffrbk_takeover_exact_noreparam_shim_v2 as runtime


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


def candidate_problem(total_budget=5000):
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
    return path, energies, forces, tangents, evaluators, budget


def completion_fixture():
    path = np.zeros((22, 3), dtype=float)
    endpoint = path.copy()
    endpoint[1:-1, 1] = -0.25
    counter_vector = np.ones(22, dtype=int)
    counter_vector[1:-1] = 2
    synchronized_counters = [{"joint_evaluations": counter_vector}]
    evaluators = [
        SimpleNamespace(counters=SimpleNamespace(joint_evaluations=1))
        for _ in range(22)
    ]

    def explicit_solve(
        path,
        energies,
        forces,
        tangents,
        evaluators,
        alpha,
        max_step,
        max_new_calls,
        *,
        inner_max_steps=10,
        stop_mode="r08",
        relative_residual=0.8,
        history_size=7,
        kkt_atol=1.0e-8,
        preconditioner_bank=None,
        active_images=None,
    ):
        del (
            path,
            tangents,
            evaluators,
            alpha,
            max_step,
            max_new_calls,
            inner_max_steps,
            stop_mode,
            relative_residual,
            history_size,
            kkt_atol,
            preconditioner_bank,
            active_images,
        )
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

    def generic_outer(*args, **kwargs):
        return explicit_solve(*args, **kwargs)

    values = {
        "path": path,
        "energies": np.zeros(22),
        "forces": np.zeros_like(path),
        "tangents": np.zeros((20, 3)),
        "evaluators": evaluators,
        "alpha": 1.0,
        "max_step": 1.0,
        "max_new_calls": 5000,
    }
    return explicit_solve, generic_outer, endpoint, values


class ExactNoReparamV2RuntimeTests(unittest.TestCase):
    def test_legacy_generic_outer_reproduces_path_keyerror(self):
        explicit, outer, _, values = completion_fixture()
        frozen = SimpleNamespace(solve_frozen_discrete_b040_band=outer)
        installed = legacy.install_exact_solve_completion_audit(frozen)
        try:
            ordered = [values[name] for name in runtime.POSITIONAL_ARGUMENTS]
            with self.assertRaisesRegex(KeyError, "path"):
                frozen.solve_frozen_discrete_b040_band(*ordered)
        finally:
            installed.restore()
        self.assertIs(frozen.solve_frozen_discrete_b040_band, outer)
        self.assertEqual(tuple(inspect.signature(explicit).parameters), (
            runtime.POSITIONAL_ARGUMENTS + runtime.KEYWORD_ONLY_ARGUMENTS
        ))

    def test_generic_outer_accepts_runner_positional_and_keyword_invocations(self):
        for invocation in ("runner-positional", "all-keyword"):
            with self.subTest(invocation=invocation):
                explicit, outer, endpoint, values = completion_fixture()
                frozen = SimpleNamespace(
                    solve_frozen_discrete_b040_band=outer
                )
                self.assertFalse(hasattr(outer, "__signature__"))
                installed = runtime.install_exact_solve_completion_audit(
                    frozen, explicit
                )
                self.assertFalse(hasattr(outer, "__signature__"))
                try:
                    if invocation == "runner-positional":
                        ordered = [
                            values[name] for name in runtime.POSITIONAL_ARGUMENTS
                        ]
                        frozen.solve_frozen_discrete_b040_band(*ordered)
                    else:
                        frozen.solve_frozen_discrete_b040_band(**values)
                finally:
                    installed.restore()
                self.assertIs(
                    frozen.solve_frozen_discrete_b040_band, outer
                )
                self.assertEqual(len(installed.events), 1)
                event = installed.events[0]
                self.assertEqual(event.exact_endpoint_nf, 40)
                np.testing.assert_array_equal(
                    event.raw_exact_endpoint_path, endpoint
                )

    def test_v2_rejects_other_argument_name_or_kind_assumptions(self):
        explicit, _, _, _ = completion_fixture()
        signature = inspect.signature(explicit)
        parameters = list(signature.parameters.values())
        parameters[0] = parameters[0].replace(name="positions")
        with self.assertRaisesRegex(
            runtime.ExactNoReparamV2SafetyError,
            "argument names/order drifted",
        ):
            runtime.validate_candidate_call_signature(
                signature.replace(parameters=parameters)
            )
        parameters = list(signature.parameters.values())
        parameters[8] = parameters[8].replace(
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        with self.assertRaisesRegex(
            runtime.ExactNoReparamV2SafetyError,
            "controls must remain keyword-only",
        ):
            runtime.validate_candidate_call_signature(
                signature.replace(parameters=parameters)
            )

    def test_candidate_selector_eta_chain_accepts_runner_positional_call(self):
        canonical_signature = inspect.signature(
            discrete.solve_frozen_discrete_b040_band
        )
        problem = candidate_problem()
        path, energies, forces, tangents, evaluators, budget = problem
        controls = legacy.grid_runtime.GridControls(
            alpha=1.0,
            residual_ratio=0.1,
            inner_max_steps=5000,
            total_internal_nf_budget=5000,
        )
        candidate_installed = None
        selector_installed = None
        eta_installed = None
        completion_installed = None
        original = discrete.solve_frozen_discrete_b040_band
        original_id = discrete.IMPLEMENTATION_ID
        with legacy.grid_runtime.configured_selector_policy(
            legacy.fixed_selector, controls
        ) as selector_module:
            candidate_installed = (
                candidate_budget.install_candidate_aware_budget_solver(
                    discrete, budget
                )
            )
            selector_installed = selector_module.install_ffrbk_selector_shim(
                discrete, "one_way_takeover"
            )
            eta_installed = legacy.selected_eta.install_frozen_eta_shim(
                discrete, 1.1
            )
            self.assertEqual(
                tuple(inspect.signature(
                    discrete.solve_frozen_discrete_b040_band
                ).parameters),
                ("args", "kwargs"),
            )
            completion_installed = (
                runtime.install_exact_solve_completion_audit(
                    discrete, canonical_signature
                )
            )
            try:
                result = discrete.solve_frozen_discrete_b040_band(
                    path,
                    energies,
                    forces,
                    tangents,
                    evaluators,
                    1.0,
                    1.0,
                    5000,
                    inner_max_steps=5000,
                    stop_mode="r08c10",
                    relative_residual=0.1,
                    history_size=7,
                    kkt_atol=0.0,
                    preconditioner_bank=IdentityBank(),
                )
                self.assertGreater(result[8]["joint_steps"], 0)
                self.assertEqual(len(completion_installed.events), 1)
                self.assertIn(
                    completion_installed.events[0].stop_reason,
                    (
                        "evaluated-relative-natural-residual",
                        "evaluated-absolute-natural-residual",
                    ),
                )
            finally:
                completion_installed.restore()
                eta_installed.restore()
                selector_installed.restore()
                candidate_installed.restore()
        self.assertIs(discrete.solve_frozen_discrete_b040_band, original)
        self.assertEqual(discrete.IMPLEMENTATION_ID, original_id)

    def test_v2_main_injects_unique_identity_and_restores_module_state(self):
        original_id = discrete.IMPLEMENTATION_ID
        original_policy = legacy.POLICY_ID
        original_installer = legacy.install_exact_solve_completion_audit
        observed = {}

        def fake_legacy_main(argv):
            observed["argv"] = list(argv)
            observed["policy"] = legacy.POLICY_ID
            observed["implementation"] = discrete.IMPLEMENTATION_ID
            observed["installer"] = legacy.install_exact_solve_completion_audit

        argv = ["v2-runtime", "--sentinel"]
        with mock.patch.object(legacy, "main", side_effect=fake_legacy_main):
            runtime.main(argv)

        self.assertEqual(observed["argv"], argv)
        self.assertEqual(observed["policy"], runtime.POLICY_ID)
        self.assertEqual(
            observed["implementation"],
            "+".join((original_id, runtime.IMPLEMENTATION_POLICY_ID)),
        )
        self.assertIsNot(observed["installer"], original_installer)
        self.assertEqual(discrete.IMPLEMENTATION_ID, original_id)
        self.assertEqual(legacy.POLICY_ID, original_policy)
        self.assertIs(
            legacy.install_exact_solve_completion_audit,
            original_installer,
        )


if __name__ == "__main__":
    unittest.main()
