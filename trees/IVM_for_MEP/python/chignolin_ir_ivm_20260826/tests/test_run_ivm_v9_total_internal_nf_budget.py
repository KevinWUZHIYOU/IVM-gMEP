from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

import chignolin_core
import frozen_discrete_b040 as discrete
import run_ivm_v9_discrete_ffrbk_takeover_grid_shim as grid_runtime
import run_ivm_v9_discrete_ffrbk_takeover_selector_shim as fixed_selector
import run_ivm_v9_total_internal_nf_budget as total_budget


class FakeBackend:
    def __init__(self):
        self.raw_joint_evaluations = 0

    def evaluate_joint(self, positions):
        self.raw_joint_evaluations += 1
        position = np.asarray(positions, dtype=float)
        return float(np.sum(position**2)), -2.0 * position


class NoQueryEvaluator:
    def __init__(self):
        self.counters = SimpleNamespace(
            joint_evaluations=0,
            energy_requests=0,
            force_requests=0,
            cache_hits=0,
            total_query_events=0,
            both_requests=0,
            budget_blocked_misses=0,
        )

    def has_cached(self, position):
        del position
        return False

    def query(self, position, property_name="both"):
        del position, property_name
        raise AssertionError("preflight attempted a physical query")


class TotalInternalNFBudgetTests(unittest.TestCase):
    def test_cache_miss_boundary_is_exact_and_preserves_counter_identities(self):
        backend = FakeBackend()
        evaluator = chignolin_core.ImageEvaluator(backend, image_index=1)
        original_query = chignolin_core.ImageEvaluator.query
        installed = total_budget.install_total_internal_nf_budget(chignolin_core, 2)
        try:
            p0 = np.asarray([0.0, 0.0, 0.0])
            p1 = np.asarray([1.0, 0.0, 0.0])
            p2 = np.asarray([2.0, 0.0, 0.0])
            evaluator.query(p0, "both")
            evaluator.query(p1, "both")
            evaluator.query(p0, "both")
            with self.assertRaises(chignolin_core.EvaluationBudgetReached):
                evaluator.query(p2, "both")
            self.assertEqual(installed.successful_internal_joint_evaluations, 2)
            self.assertEqual(installed.blocked_internal_misses, 1)
            self.assertEqual(backend.raw_joint_evaluations, 2)
            self.assertEqual(evaluator.counters.joint_evaluations, 2)
            self.assertEqual(evaluator.counters.cache_hits, 1)
            self.assertEqual(evaluator.counters.budget_blocked_misses, 1)
            self.assertEqual(
                evaluator.counters.cache_hits
                + evaluator.counters.joint_evaluations
                + evaluator.counters.budget_blocked_misses,
                evaluator.counters.total_query_events,
            )
        finally:
            installed.restore()
        self.assertIs(chignolin_core.ImageEvaluator.query, original_query)

    @staticmethod
    def _counter_vectors(costs):
        counters = np.zeros((len(costs), 22), dtype=int)
        for row, cost in enumerate(costs):
            quotient, remainder = divmod(int(cost), 20)
            counters[row, 1:21] = quotient
            counters[row, 1 : 1 + remainder] += 1
            counters[row, 0] = 1
            counters[row, -1] = 1
        return counters

    def test_partial_tail_metadata_uses_declared_completeness_policy(self):
        installed = total_budget.InstalledTotalInternalNFBudget(
            ala_core_module=chignolin_core,
            budget=50,
            original_query=chignolin_core.ImageEvaluator.query,
            completeness_policy=(
                total_budget.COMPLETENESS_INITIAL_OR_REPARAM_COMMITTED
            ),
            successful_internal_joint_evaluations=50,
        )
        arrays = {
            "total_internal_joint_evaluations": np.asarray([20, 40, 50]),
            "counter_vectors": self._counter_vectors([20, 40, 50]),
            "reparam_committed": np.asarray([False, True, False]),
        }
        metadata = installed.result_metadata(arrays)
        self.assertEqual(metadata["total_internal_nf_last_complete_state_nf"], 40)
        self.assertEqual(
            metadata["total_internal_nf_terminal_uncommitted_charged_tail_nf"],
            10,
        )
        self.assertTrue(metadata["total_internal_nf_terminal_partial_tail_present"])
        self.assertTrue(
            metadata["total_internal_nf_terminal_partial_tail_quarantined"]
        )
        self.assertEqual(metadata["total_internal_nf_remaining"], 0)
        self.assertEqual(
            metadata["total_internal_nf_budget_stop_reason"],
            "total_internal_joint_budget_exhausted",
        )

        installed.completeness_policy = (
            total_budget.COMPLETENESS_ALL_RECORDED_STATES
        )
        metadata = installed.result_metadata(arrays)
        self.assertEqual(metadata["total_internal_nf_last_complete_state_nf"], 50)
        self.assertFalse(metadata["total_internal_nf_terminal_partial_tail_present"])

        installed.successful_internal_joint_evaluations = 49
        installed.full_band_preflight_zero_step_stops = 1
        installed.complete_band_budget_stop_count = 1
        short_arrays = {
            "total_internal_joint_evaluations": np.asarray([49]),
            "counter_vectors": self._counter_vectors([49]),
            "reparam_committed": np.asarray([False]),
        }
        metadata = installed.result_metadata(short_arrays)
        self.assertEqual(metadata["total_internal_nf_remaining"], 1)
        self.assertEqual(
            metadata["total_internal_nf_budget_stop_reason"],
            "total_internal_joint_budget_insufficient_for_full_band",
        )

    def test_keyed_mask_quarantines_charged_terminal_tail(self):
        installed = total_budget.InstalledTotalInternalNFBudget(
            ala_core_module=chignolin_core,
            budget=80,
            original_query=chignolin_core.ImageEvaluator.query,
            completeness_policy=total_budget.COMPLETENESS_KEYED_MASK,
            completeness_mask_key="exact_mask",
            successful_internal_joint_evaluations=80,
        )
        arrays = {
            "total_internal_joint_evaluations": np.asarray([20, 40, 60, 80]),
            "counter_vectors": self._counter_vectors([20, 40, 60, 80]),
            "reparam_committed": np.zeros(4, dtype=bool),
            "exact_mask": np.asarray([True, False, True, False]),
        }
        metadata = installed.result_metadata(arrays)
        self.assertEqual(metadata["total_internal_nf_last_complete_state_index"], 2)
        self.assertEqual(metadata["total_internal_nf_last_complete_state_nf"], 60)
        self.assertEqual(
            metadata["total_internal_nf_terminal_uncommitted_charged_tail_nf"],
            20,
        )
        self.assertTrue(metadata["total_internal_nf_terminal_partial_tail_present"])
        self.assertEqual(
            metadata["total_internal_nf_complete_state_mask_key"], "exact_mask"
        )

    def test_less_than_one_band_remaining_returns_without_query(self):
        budget_state = SimpleNamespace(
            ala_core_module=chignolin_core,
            budget=5000,
            successful_internal_joint_evaluations=4981,
            full_band_preflight_calls=0,
            full_band_preflight_crops=0,
            full_band_preflight_zero_step_stops=0,
            complete_band_budget_stop_count=0,
        )
        original = discrete.solve_frozen_discrete_b040_band
        installed = grid_runtime.install_full_band_budget_preflight(
            discrete, budget_state
        )
        try:
            path = np.zeros((22, 3), dtype=float)
            path[:, 0] = np.linspace(-1.0, 1.0, 22)
            energies = np.zeros(22, dtype=float)
            forces = np.zeros_like(path)
            forces[1:-1, 1] = -1.0
            tangents = np.zeros((20, 3), dtype=float)
            tangents[:, 0] = 1.0
            evaluators = [NoQueryEvaluator() for _ in range(22)]
            result = discrete.solve_frozen_discrete_b040_band(
                path,
                energies,
                forces,
                tangents,
                evaluators,
                1.0,
                1.0,
                10,
                inner_max_steps=10,
                stop_mode="r08c10",
                relative_residual=0.8,
                history_size=7,
                kkt_atol=0.0,
                preconditioner_bank=object(),
            )
            self.assertEqual(result[8]["joint_steps"], 0)
            self.assertEqual(result[8]["paid_trials"], 0)
            self.assertEqual(result[8]["stop_reason"], "per-image-new-call-limit")
            self.assertEqual(budget_state.full_band_preflight_calls, 1)
            self.assertEqual(budget_state.full_band_preflight_crops, 1)
            self.assertEqual(budget_state.full_band_preflight_zero_step_stops, 1)
            self.assertEqual(budget_state.complete_band_budget_stop_count, 1)
        finally:
            installed.restore()
        self.assertIs(discrete.solve_frozen_discrete_b040_band, original)

    def test_parametric_selector_context_is_injective_and_restores_old_shim(self):
        old_alpha = fixed_selector.FIXED_ALPHA
        old_cap = fixed_selector.FIXED_INNER_MAX_STEPS
        old_builder = fixed_selector.selector_implementation_id
        controls = grid_runtime.GridControls(1.0e6, 0.1, 250, 5000)
        with grid_runtime.configured_selector_policy(
            fixed_selector, controls
        ) as configured:
            self.assertEqual(configured.FIXED_ALPHA, 1.0e6)
            self.assertEqual(configured.FIXED_RESIDUAL_RATIO, 0.1)
            self.assertEqual(configured.FIXED_INNER_MAX_STEPS, 250)
            identifier = configured.selector_implementation_id(
                "base", "one_way_takeover"
            )
            self.assertIn(grid_runtime.POLICY_ID, identifier)
            self.assertIn("c250", identifier)
            self.assertIn("totalinternalnf5000", identifier)
        self.assertEqual(fixed_selector.FIXED_ALPHA, old_alpha)
        self.assertEqual(fixed_selector.FIXED_INNER_MAX_STEPS, old_cap)
        self.assertIs(fixed_selector.selector_implementation_id, old_builder)


if __name__ == "__main__":
    unittest.main()
