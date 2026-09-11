from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

import frozen_discrete_b040 as discrete
import run_ivm_v9_discrete_ffrbk_takeover_selector_shim as takeover
import run_ivm_v9_discrete_selected_eta_shim as eta_shim


BASE_IMPLEMENTATION_ID = "chignolin-frozen-discrete-ffrbk-dsel-m7-evaluated-v1"


class MockEvaluator:
    def __init__(self, stiffness=0.05):
        self.stiffness = float(stiffness)
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
        self.counters.energy_requests += int(property_name in {"energy", "both"})
        self.counters.force_requests += int(property_name in {"force", "both"})
        self.counters.both_requests += int(property_name == "both")
        key = self._key(position)
        if key in self.cache:
            self.counters.cache_hits += 1
            energy, force = self.cache[key]
        else:
            position = np.asarray(position, dtype=np.float64)
            energy = 0.5 * self.stiffness * float(position[1] ** 2)
            force = np.zeros_like(position)
            force[1] = -self.stiffness * position[1]
            self.cache[key] = (energy, force.copy())
            self.counters.joint_evaluations += 1
        if property_name == "energy":
            return energy
        if property_name == "force":
            return force.copy()
        return energy, force.copy()


class IdentityBank:
    def __init__(self):
        self.calls = 0

    def solve_normal(
        self, image_index, vector, tangent, diagonal_shift=0.0
    ):
        del image_index, tangent, diagonal_shift
        self.calls += 1
        return np.asarray(vector, dtype=np.float64)


def one_image_problem(stiffness=0.05):
    path = np.asarray(
        [(-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)],
        dtype=np.float64,
    )
    evaluators = [MockEvaluator(stiffness) for _ in range(3)]
    energies, forces = discrete.evaluate_path(path, evaluators)
    tangents = np.asarray([(1.0, 0.0, 0.0)], dtype=np.float64)
    return path, energies, forces, tangents, evaluators


def exact_runner_argv():
    return [
        "shim.py",
        "--solver",
        "state_ivm",
        "--frozen-subproblem",
        "discrete_b040",
        "--alpha",
        "15",
        "--alpha-mode",
        "fixed",
        "--max-step",
        "1",
        "--history",
        "7",
        "--discrete-history",
        "7",
        "--discrete-inner-max-steps",
        "10",
        "--discrete-stop-mode",
        "r08c10",
        "--discrete-residual-ratio",
        "0.8",
        "--plbfgs-kkt-atol",
        "0",
        "--reparam-mode",
        "state_curve",
        "--state-curve-direct-objective-acceptance",
        "per_image",
        "--state-globalization",
        "iperp_monotone",
        "--state-trust-fraction",
        "2",
        "--stop-complete-iperp",
        "-1",
        "--state-curve-direct-solve",
    ]


def proposal(label, decrease, *, valid=True, step_value=None):
    if step_value is None:
        step_value = -float(decrease) if valid else 0.0
    return {
        "label": label,
        "valid": bool(valid),
        "error": "none" if valid else "invalid-for-test",
        "raw": np.asarray([[0.0, step_value, 0.0]]) if valid else None,
        "step": np.asarray([[0.0, step_value, 0.0]]) if valid else None,
        "raw_decrease": float(decrease) if valid else 0.0,
        "capped_decrease": float(decrease) if valid else 0.0,
        "cap_scale": 1.0 if valid else 0.0,
    }


class FFrbkTakeoverSelectorShimTests(unittest.TestCase):
    def tearDown(self):
        self.assertEqual(discrete.IMPLEMENTATION_ID, BASE_IMPLEMENTATION_ID)

    def test_cli_fixed_controls_and_unique_composed_ids(self):
        argv = exact_runner_argv() + [
            takeover.MODE_FLAG,
            "one_way_takeover",
            eta_shim.ETA_FLAG,
            "1.1",
        ]
        mode, eta, cleaned = takeover.extract_runtime_arguments(argv)
        self.assertEqual(mode, "one_way_takeover")
        self.assertEqual(eta, 1.1)
        self.assertEqual(cleaned, exact_runner_argv())
        takeover.validate_runner_argv(cleaned)
        first = takeover.final_implementation_id("base", "one_way_takeover")
        second = takeover.final_implementation_id(
            "base", "forced_operator_lbfgs_selection"
        )
        self.assertNotEqual(first, second)
        self.assertIn(takeover.RUNTIME_POLICY_ID, first)
        self.assertIn(eta_shim.SHIM_POLICY_ID, first)

        for bad_mode in ("takeover", "", "LBFGS", "operator_lbfgs_only"):
            bad = list(argv)
            bad[bad.index("one_way_takeover")] = bad_mode
            with self.assertRaises(ValueError):
                takeover.extract_runtime_arguments(bad)
        bad_eta = list(argv)
        bad_eta[bad_eta.index("1.1")] = "1.0"
        with self.assertRaisesRegex(ValueError, "eta=1.1"):
            takeover.extract_runtime_arguments(bad_eta)
        for flag, value in (
            ("--alpha", "12.5"),
            ("--max-step", "0.8"),
            ("--discrete-stop-mode", "r08"),
            ("--plbfgs-kkt-atol", "1e-8"),
            ("--reparam-mode", "polyline"),
        ):
            bad = exact_runner_argv()
            bad[bad.index(flag) + 1] = value
            with self.assertRaises(ValueError):
                takeover.validate_runner_argv(bad)

    def test_one_way_strict_win_latches_then_invalid_lbfgs_fails_over(self):
        state = takeover.SelectorState()
        tie_h0 = proposal("H0", 2.0, step_value=-0.2)
        tie_lbfgs = proposal("LBFGS", 2.0, step_value=-0.2)
        selected, event = takeover.choose_direction(
            "one_way_takeover",
            state,
            tie_h0,
            tie_lbfgs,
            discrete.select_direction,
        )
        self.assertIs(selected, tie_h0)
        self.assertFalse(state.latched)
        self.assertTrue(event.empty_history_tie)
        self.assertFalse(event.strict_win_trigger)

        h0 = proposal("H0", 3.0)
        winning_lbfgs = proposal("LBFGS", 4.0)
        selected, event = takeover.choose_direction(
            "one_way_takeover",
            state,
            h0,
            winning_lbfgs,
            discrete.select_direction,
        )
        self.assertIs(selected, winning_lbfgs)
        self.assertTrue(event.strict_win_trigger)
        self.assertFalse(event.latched_before)
        self.assertTrue(event.latched_after)

        # Once latched, valid L-BFGS wins even when H0 has larger decrease.
        selected, event = takeover.choose_direction(
            "one_way_takeover",
            state,
            proposal("H0", 10.0),
            proposal("LBFGS", 1.0),
            discrete.select_direction,
        )
        self.assertEqual(selected["label"], "LBFGS")
        self.assertTrue(event.forced_lbfgs)
        self.assertTrue(event.latched_before)

        selected, event = takeover.choose_direction(
            "one_way_takeover",
            state,
            proposal("H0", 1.0),
            proposal("LBFGS", 0.0, valid=False),
            discrete.select_direction,
        )
        self.assertEqual(selected["label"], "H0")
        self.assertTrue(event.h0_failover)
        self.assertTrue(state.latched)

        # A new frozen solver owns a genuinely fresh latch.
        fresh = takeover.SelectorState()
        selected, event = takeover.choose_direction(
            "one_way_takeover",
            fresh,
            proposal("H0", 5.0),
            proposal("LBFGS", 1.0),
            discrete.select_direction,
        )
        self.assertEqual(selected["label"], "H0")
        self.assertFalse(event.latched_before)
        self.assertFalse(event.latched_after)

    def test_forced_operator_selection_owns_tie_and_has_h0_failover(self):
        state = takeover.SelectorState()
        h0 = proposal("H0", 2.0, step_value=-0.2)
        lbfgs = proposal("LBFGS", 2.0, step_value=-0.2)
        selected, event = takeover.choose_direction(
            "forced_operator_lbfgs_selection",
            state,
            h0,
            lbfgs,
            discrete.select_direction,
        )
        self.assertIs(selected, lbfgs)
        self.assertTrue(event.empty_history_tie)
        self.assertTrue(event.forced_lbfgs)
        self.assertFalse(state.latched)

        selected, event = takeover.choose_direction(
            "forced_operator_lbfgs_selection",
            state,
            proposal("H0", 100.0),
            proposal("LBFGS", 1.0),
            discrete.select_direction,
        )
        self.assertEqual(selected["label"], "LBFGS")
        self.assertTrue(event.forced_lbfgs)
        selected, event = takeover.choose_direction(
            "forced_operator_lbfgs_selection",
            state,
            proposal("H0", 1.0),
            proposal("LBFGS", 0.0, valid=False),
            discrete.select_direction,
        )
        self.assertEqual(selected["label"], "H0")
        self.assertTrue(event.h0_failover)

    def _install_composed(self, mode):
        selector = takeover.install_ffrbk_selector_shim(discrete, mode)
        eta = eta_shim.install_frozen_eta_shim(discrete, 1.1)
        self.assertEqual(
            eta.implementation_id,
            takeover.final_implementation_id(BASE_IMPLEMENTATION_ID, mode),
        )
        return selector, eta

    def _run_composed_solver(self, bank):
        path, energies, forces, tangents, evaluators = one_image_problem()
        before = evaluators[1].counters.joint_evaluations
        result = discrete.solve_frozen_discrete_b040_band(
            path,
            energies,
            forces,
            tangents,
            evaluators,
            15.0,
            1.0,
            10,
            inner_max_steps=10,
            stop_mode="r08c10",
            relative_residual=0.8,
            history_size=7,
            kkt_atol=0.0,
            preconditioner_bank=bank,
        )
        after = evaluators[1].counters.joint_evaluations
        self.assertEqual(after - before, result[8]["joint_steps"])
        self.assertEqual(int(result[3][0]), result[8]["joint_steps"])
        return result

    def test_real_frozen_solver_preserves_operator_eta_order_and_fresh_latch(self):
        selector, eta = self._install_composed("one_way_takeover")
        try:
            first_bank = IdentityBank()
            first = self._run_composed_solver(first_bank)
            second_bank = IdentityBank()
            second = self._run_composed_solver(second_bank)
            self.assertGreater(first_bank.calls, 0)
            self.assertGreater(second_bank.calls, 0)
            self.assertEqual(selector.audit.fixed_solver_calls, 2)
            self.assertEqual(selector.audit.latch_reset_count, 2)
            self.assertEqual(
                selector.audit.apply_h0_inverse_preconditioner_none_calls, 0
            )
            self.assertGreater(selector.audit.joint_lbfgs_direction_calls, 0)
            self.assertEqual(len(selector.audit.contexts), 2)
            self.assertFalse(selector.audit.contexts[0].events[0].latched_before)
            self.assertFalse(selector.audit.contexts[1].events[0].latched_before)
            self.assertTrue(
                any(event.strict_win_trigger for event in selector.audit.committed)
            )
            np.testing.assert_array_equal(
                first[8]["direction_selection"], ["H0", "LBFGS"]
            )
            np.testing.assert_array_equal(
                second[8]["direction_selection"], ["H0", "LBFGS"]
            )
            self.assertTrue(
                np.allclose(
                    [e.requested_multiplier for e in eta.audit.committed], 1.1
                )
            )
        finally:
            eta.restore()
            selector.restore()

    def test_operator_mode_composed_metadata_is_aligned_and_no_extra_nf(self):
        selector, eta = self._install_composed(
            "forced_operator_lbfgs_selection"
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "result.npz"

                def save_npz(path, **arrays):
                    np.savez(path, **arrays)

                fake_runner = SimpleNamespace(
                    DISCRETE_B040_IMPLEMENTATION_ID=discrete.IMPLEMENTATION_ID,
                    solve_frozen_discrete_b040_band=(
                        discrete.solve_frozen_discrete_b040_band
                    ),
                    atomic_save_npz=save_npz,
                )
                eta.attach_runner(fake_runner)
                selector.attach_runner(fake_runner, eta.implementation_id)
                path, energies, forces, tangents, evaluators = one_image_problem()
                result = fake_runner.solve_frozen_discrete_b040_band(
                    path,
                    energies,
                    forces,
                    tangents,
                    evaluators,
                    15.0,
                    1.0,
                    10,
                    inner_max_steps=10,
                    stop_mode="r08c10",
                    relative_residual=0.8,
                    history_size=7,
                    kkt_atol=0.0,
                    preconditioner_bank=IdentityBank(),
                )
                self.assertGreaterEqual(result[8]["joint_steps"], 1)
                selection = np.asarray(
                    result[8]["direction_selection"], dtype="U8"
                )
                self.assertTrue(np.all(selection == "LBFGS"))
                fake_runner.atomic_save_npz(
                    destination,
                    frozen_subproblem=np.array("discrete_b040"),
                    discrete_implementation_id=np.array(eta.implementation_id),
                    alpha=np.array(15.0),
                    alpha_mode=np.array("fixed"),
                    active_alpha=np.asarray([15.0, 15.0]),
                    discrete_inner_max_steps=np.array(10),
                    discrete_stop_mode=np.array("r08c10"),
                    discrete_residual_ratio=np.array(0.8),
                    discrete_history=np.array(7),
                    plbfgs_kkt_atol=np.array(0.0),
                    reparam_mode=np.array("state_curve"),
                    state_curve_direct_objective_acceptance=np.array("per_image"),
                    discrete_audit_selection=selection,
                )
                with np.load(destination, allow_pickle=False) as saved:
                    self.assertEqual(
                        saved["discrete_ffrbk_selector_mode"].item(),
                        "forced_operator_lbfgs_selection",
                    )
                    self.assertTrue(
                        bool(saved["discrete_ffrbk_selector_operator_preserved"])
                    )
                    self.assertTrue(
                        bool(
                            saved[
                                "discrete_ffrbk_selector_h0_shadow_constructed"
                            ]
                        )
                    )
                    self.assertTrue(
                        bool(
                            saved[
                                "discrete_ffrbk_selector_selector_only_change"
                            ]
                        )
                    )
                    self.assertFalse(
                        bool(
                            saved[
                                "discrete_ffrbk_selector_true_lbfgs_only_construction"
                            ]
                        )
                    )
                    self.assertTrue(
                        bool(
                            saved[
                                "discrete_ffrbk_selector_invalid_lbfgs_h0_fallback_enabled"
                            ]
                        )
                    )
                    self.assertTrue(
                        bool(
                            saved[
                                "discrete_ffrbk_selector_post_select_eta_composed"
                            ]
                        )
                    )
                    self.assertEqual(
                        int(
                            saved[
                                "discrete_ffrbk_selector_extra_physical_evaluations"
                            ]
                        ),
                        0,
                    )
                    self.assertFalse(
                        bool(
                            saved[
                                "discrete_ffrbk_selector_energy_backtracking"
                            ]
                        )
                    )
                    np.testing.assert_array_equal(
                        saved["discrete_ffrbk_selector_audit_selection"],
                        selection,
                    )
                    self.assertTrue(
                        bool(
                            saved[
                                "discrete_ffrbk_selector_audit_empty_history_tie"
                            ][0]
                        )
                    )
                    self.assertTrue(
                        np.all(
                            saved[
                                "discrete_ffrbk_selector_audit_forced_lbfgs"
                            ]
                        )
                    )
                    np.testing.assert_array_equal(
                        saved["discrete_step_multiplier_audit_selection"],
                        selection,
                    )
        finally:
            eta.restore()
            selector.restore()

    def test_missing_ffrbk_bank_fails_closed_and_restore_is_exact(self):
        original_select = discrete.select_direction
        original_solve = discrete.solve_frozen_discrete_b040_band
        original_apply = discrete.apply_block_h0_inverse
        selector = takeover.install_ffrbk_selector_shim(
            discrete, "one_way_takeover"
        )
        try:
            path, energies, forces, tangents, evaluators = one_image_problem()
            with self.assertRaisesRegex(
                takeover.FFrbkSelectorSafetyError, "preconditioner bank"
            ):
                discrete.solve_frozen_discrete_b040_band(
                    path,
                    energies,
                    forces,
                    tangents,
                    evaluators,
                    15.0,
                    1.0,
                    10,
                    inner_max_steps=10,
                    stop_mode="r08c10",
                    relative_residual=0.8,
                    history_size=7,
                    kkt_atol=0.0,
                    preconditioner_bank=None,
                )
        finally:
            selector.restore()
        self.assertIs(discrete.select_direction, original_select)
        self.assertIs(discrete.solve_frozen_discrete_b040_band, original_solve)
        self.assertIs(discrete.apply_block_h0_inverse, original_apply)

    def test_composed_fresh_import_order_restores_authoritative_base(self):
        original_solve = discrete.solve_frozen_discrete_b040_band
        selector, eta = self._install_composed(
            "forced_operator_lbfgs_selection"
        )
        # This mirrors main(): the runner imports only after both frozen-module
        # patches and therefore captures the final composed pair by value.
        fresh_runner = SimpleNamespace(
            DISCRETE_B040_IMPLEMENTATION_ID=discrete.IMPLEMENTATION_ID,
            solve_frozen_discrete_b040_band=(
                discrete.solve_frozen_discrete_b040_band
            ),
            atomic_save_npz=lambda path, **arrays: None,
        )
        self.assertEqual(
            fresh_runner.DISCRETE_B040_IMPLEMENTATION_ID,
            takeover.final_implementation_id(
                BASE_IMPLEMENTATION_ID,
                "forced_operator_lbfgs_selection",
            ),
        )
        self.assertIs(
            fresh_runner.solve_frozen_discrete_b040_band,
            discrete.solve_frozen_discrete_b040_band,
        )
        try:
            eta.attach_runner(fresh_runner)
            selector.attach_runner(fresh_runner, eta.implementation_id)
        finally:
            # The outer eta layer must be removed before the selector layer.
            eta.restore()
            selector.restore()
        self.assertEqual(
            fresh_runner.DISCRETE_B040_IMPLEMENTATION_ID,
            BASE_IMPLEMENTATION_ID,
        )
        self.assertIs(fresh_runner.solve_frozen_discrete_b040_band, original_solve)
        self.assertIs(discrete.solve_frozen_discrete_b040_band, original_solve)


if __name__ == "__main__":
    unittest.main()
