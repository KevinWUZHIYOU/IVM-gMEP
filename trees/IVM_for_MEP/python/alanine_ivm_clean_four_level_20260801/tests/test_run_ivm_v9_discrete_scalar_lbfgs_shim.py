from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

import frozen_discrete_b040 as discrete
import run_ivm_v9_discrete_scalar_lbfgs_shim as scalar
import run_ivm_v9_spring_projected_production_20260730 as legacy_runner


BASE_IMPLEMENTATION_ID = "ala-frozen-discrete-ffrbk-dsel-m7-evaluated-v1"


class MockEvaluator:
    def __init__(self, stiffness=0.05, potential=None):
        self.stiffness = float(stiffness)
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
        self.counters.energy_requests += int(property_name in {"energy", "both"})
        self.counters.force_requests += int(property_name in {"force", "both"})
        self.counters.both_requests += int(property_name == "both")
        key = self._key(position)
        if key in self.cache:
            self.counters.cache_hits += 1
            energy, force = self.cache[key]
        else:
            position = np.asarray(position, dtype=np.float64)
            if self.potential is None:
                energy = 0.5 * self.stiffness * float(position[1] ** 2)
                force = np.zeros_like(position)
                force[1] = -self.stiffness * position[1]
            else:
                energy, force = self.potential(position)
                energy = float(energy)
                force = np.asarray(force, dtype=np.float64)
            self.cache[key] = (energy, force.copy())
            self.counters.joint_evaluations += 1
        if property_name == "energy":
            return energy
        if property_name == "force":
            return force.copy()
        return energy, force.copy()


def one_image_problem(stiffness=0.05, potential=None):
    path = np.asarray(
        [(-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)],
        dtype=np.float64,
    )
    evaluators = [MockEvaluator(stiffness, potential) for _ in range(3)]
    energies, forces = discrete.evaluate_path(path, evaluators)
    tangents = np.asarray([(1.0, 0.0, 0.0)], dtype=np.float64)
    return path, energies, forces, tangents, evaluators


def run_shared(mode, *, stiffness=0.05, preconditioner_bank=None, **updates):
    max_step = float(updates.pop("test_max_step", 1.0))
    potential = updates.pop("test_potential", None)
    path, energies, forces, tangents, evaluators = one_image_problem(
        stiffness, potential
    )
    options = {
        "inner_max_steps": 10,
        "stop_mode": "r08c10",
        "relative_residual": 0.8,
        "history_size": 7,
        "kkt_atol": 0.0,
        "preconditioner_bank": preconditioner_bank,
        "mode": mode,
        "eta": 1.1,
        "implementation_id": "test-implementation",
        "discrete_module": discrete,
    }
    options.update(updates)
    return scalar.solve_frozen_discrete_scalar_lbfgs_band(
        path,
        energies,
        forces,
        tangents,
        evaluators,
        15.0,
        max_step,
        10,
        **options,
    )


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
    ]


class FakeBank:
    def __init__(self, backend, *args, **kwargs):
        del args, kwargs
        self.backend = backend
        self.builds = 0

    def build(self, path):
        np.asarray(path, dtype=np.float64)
        self.builds += 1

    def solve_normal(self, image_index, vector, tangent, diagonal_shift=0.0):
        del image_index, tangent, diagonal_shift
        return np.asarray(vector, dtype=np.float64)


class ScalarLBFGSShimTests(unittest.TestCase):
    def tearDown(self):
        self.assertEqual(discrete.IMPLEMENTATION_ID, BASE_IMPLEMENTATION_ID)

    def test_cli_modes_fixed_eta_unique_ids_and_protocol_fail_closed(self):
        argv = exact_runner_argv() + [
            scalar.MODE_FLAG,
            "scalar_selector",
            "--discrete-step-multiplier-eta",
            "1.1",
        ]
        mode, eta, cleaned = scalar.extract_runtime_arguments(argv)
        self.assertEqual(mode, "scalar_selector")
        self.assertEqual(eta, 1.1)
        self.assertEqual(cleaned, exact_runner_argv())
        scalar.validate_runner_argv(cleaned)
        self.assertNotEqual(
            scalar.shim_implementation_id("base", "scalar_selector"),
            scalar.shim_implementation_id("base", "direct_lbfgs_only"),
        )
        self.assertIn(
            scalar.SCALAR_INITIAL_INVERSE_ID,
            scalar.shim_implementation_id("base", "scalar_selector"),
        )
        self.assertIn(
            "r08c10-kkt0",
            scalar.shim_implementation_id("base", "scalar_selector"),
        )

        for bad_mode in ("direct", "", "H0"):
            bad = list(argv)
            bad[bad.index("scalar_selector")] = bad_mode
            with self.assertRaises(ValueError):
                scalar.extract_runtime_arguments(bad)
        bad_eta = list(argv)
        bad_eta[bad_eta.index("1.1")] = "1.0"
        with self.assertRaisesRegex(ValueError, "eta=1.1"):
            scalar.extract_runtime_arguments(bad_eta)
        for flag, bad_value in (
            ("--alpha", "12.5"),
            ("--alpha-mode", "adaptive"),
            ("--discrete-history", "6"),
            ("--plbfgs-kkt-atol", "1e-8"),
        ):
            bad = exact_runner_argv()
            bad[bad.index(flag) + 1] = bad_value
            with self.assertRaises(ValueError):
                scalar.validate_runner_argv(bad)

    def test_identity_first_and_nonempty_two_loop_match_legacy_scalar_formula(self):
        gradient = np.asarray([(0.0, 2.0, -0.5, 0.0, 0.4, 0.2)])
        tangent = np.asarray([(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)])
        empty_direction, empty_scale = scalar.direct_scalar_lbfgs_direction(
            gradient, [], [], tangent, discrete
        )
        np.testing.assert_array_equal(
            empty_direction, -discrete.project_normal(gradient, tangent)
        )
        self.assertEqual(empty_scale.gamma, 1.0)
        self.assertFalse(empty_scale.floor_applied)

        s_value = np.asarray([(0.0, 0.3, 0.1, 0.0, -0.2, 0.05)])
        y_value = np.asarray([(0.0, 0.9, 0.2, 0.0, -0.3, 0.1)])
        direction, scale = scalar.direct_scalar_lbfgs_direction(
            gradient, [s_value], [y_value], tangent, discrete
        )
        legacy = legacy_runner._projected_lbfgs_direction(
            gradient[0],
            tangent[0],
            [(s_value[0], y_value[0])],
            initial_inverse_scale=15.0,
        )
        np.testing.assert_allclose(direction[0], legacy, rtol=0.0, atol=0.0)
        raw_gamma = float(np.sum(s_value * y_value) / np.sum(y_value * y_value))
        self.assertEqual(scale.raw_gamma, raw_gamma)
        self.assertEqual(scale.gamma, max(raw_gamma, 1.0e-12))
        self.assertEqual(scale.source, "last-sy-over-yy-with-1e-12-floor")

    def test_gamma_is_one_joint_band_scalar_not_per_image_scalars(self):
        gradient = np.asarray(
            [(0.0, 1.0, 0.0), (0.0, -0.4, 0.2)], dtype=np.float64
        )
        tangents = np.asarray(
            [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)], dtype=np.float64
        )
        s_value = np.asarray(
            [(0.0, 2.0, 0.0), (0.0, 0.25, 0.5)], dtype=np.float64
        )
        y_value = np.asarray(
            [(0.0, 4.0, 0.0), (0.0, 1.5, 0.25)], dtype=np.float64
        )
        _, scale = scalar.direct_scalar_lbfgs_direction(
            gradient, [s_value], [y_value], tangents, discrete
        )
        expected_joint = float(np.sum(s_value * y_value) / np.sum(y_value**2))
        per_image = np.sum(s_value * y_value, axis=1) / np.sum(
            y_value**2, axis=1
        )
        self.assertEqual(scale.raw_gamma, expected_joint)
        self.assertFalse(np.any(per_image == expected_joint))

    def test_structural_history_corruption_fails_closed(self):
        gradient = np.asarray([(0.0, 1.0, 0.0)])
        tangent = np.asarray([(1.0, 0.0, 0.0)])
        with self.assertRaisesRegex(
            scalar.ScalarLBFGSSafetyError, "history lengths"
        ):
            scalar.direct_scalar_lbfgs_direction(
                gradient,
                [np.asarray([(0.0, 1.0, 0.0)])],
                [],
                tangent,
                discrete,
            )
        with self.assertRaisesRegex(
            scalar.ScalarLBFGSSafetyError, "pair shape"
        ):
            scalar.direct_scalar_lbfgs_direction(
                gradient,
                [np.asarray([0.0, 1.0, 0.0])],
                [np.asarray([0.0, 1.0, 0.0])],
                tangent,
                discrete,
            )

    def test_modes_share_first_step_but_only_direct_mode_has_no_selector(self):
        selector = run_shared("scalar_selector")[8]
        direct = run_shared("direct_lbfgs_only")[8]
        self.assertEqual(selector["joint_steps"], 2)
        self.assertEqual(direct["joint_steps"], 2)
        self.assertEqual(selector["direction_selection"][0], "H0")
        self.assertTrue(np.all(direct["direction_selection"] == "LBFGS"))
        self.assertTrue(np.all(selector["scalar_lbfgs_h0_constructed"]))
        self.assertTrue(np.all(selector["scalar_lbfgs_selector_used"]))
        self.assertFalse(np.any(direct["scalar_lbfgs_h0_constructed"]))
        self.assertFalse(np.any(direct["scalar_lbfgs_selector_used"]))
        self.assertEqual(selector["scalar_lbfgs_initial_inverse_scale"][0], 1.0)
        self.assertEqual(direct["scalar_lbfgs_initial_inverse_scale"][0], 1.0)
        self.assertGreater(
            direct["scalar_lbfgs_initial_inverse_scale"][1], 1.0
        )
        self.assertEqual(
            selector["stop_reason"], "evaluated-relative-natural-residual"
        )
        self.assertEqual(direct["stop_reason"], selector["stop_reason"])
        self.assertFalse(selector["scalar_lbfgs_energy_backtracking"])
        self.assertFalse(direct["scalar_lbfgs_absolute_kkt_stop_enabled"])

    def test_strict_relative_boundary_and_exact_ten_step_stop(self):
        # For this one-dimensional quadratic and identity first step,
        # R1/R0 = 1 - eta*k - eta/alpha.  Exercise both sides of 0.8.
        stiffness_079 = (1.0 - 1.1 / 15.0 - 0.79) / 1.1
        stiffness_081 = (1.0 - 1.1 / 15.0 - 0.81) / 1.1
        below = run_shared(
            "direct_lbfgs_only", stiffness=stiffness_079
        )[8]
        above = run_shared(
            "direct_lbfgs_only", stiffness=stiffness_081
        )[8]
        self.assertEqual(below["joint_steps"], 1)
        self.assertLessEqual(below["global_relative_residual"], 0.8)
        self.assertGreater(above["global_relative_residual"], 0.0)
        self.assertGreater(above["joint_steps"], 1)

        def constant_proximal_residual(position):
            displacement = float(position[1] - 1.0)
            physical_gradient = 0.05 - displacement / 15.0
            energy = 0.05 * position[1] - displacement**2 / 30.0
            force = np.zeros_like(position)
            force[1] = -physical_gradient
            return energy, force

        capped = run_shared(
            "direct_lbfgs_only", test_potential=constant_proximal_residual
        )[8]
        self.assertEqual(capped["joint_steps"], 10)
        self.assertEqual(capped["stop_reason"], "evaluated-inner-max-steps")
        self.assertAlmostEqual(capped["global_relative_residual"], 1.0)
        self.assertFalse(capped["global_converged"])

    def test_eta_recap_curvature_uses_actual_executed_step(self):
        result = run_shared(
            "direct_lbfgs_only", stiffness=0.05, test_max_step=0.052
        )
        report = result[8]
        self.assertGreaterEqual(report["joint_steps"], 2)
        self.assertTrue(report["scalar_lbfgs_cap_hit"][0])
        self.assertAlmostEqual(
            report["scalar_lbfgs_base_max_atomic_displacement"][0], 0.05
        )
        self.assertAlmostEqual(
            report["scalar_lbfgs_requested_max_atomic_displacement"][0],
            0.055,
        )
        self.assertAlmostEqual(
            report["scalar_lbfgs_applied_max_atomic_displacement"][0],
            0.052,
        )

        initial_path, _, initial_forces, tangents, _ = one_image_problem(0.05)
        first_path = result[4][0]
        first_forces = result[6][0]
        first_displacement = first_path[1:-1] - initial_path[1:-1]
        initial_residual = discrete.project_normal(
            -initial_forces[1:-1], tangents
        )
        first_residual = discrete.project_normal(
            -first_forces[1:-1] + first_displacement / 15.0,
            tangents,
        )
        y_value = first_residual - initial_residual
        expected_sy = float(np.sum(first_displacement * y_value))
        expected_yy = float(np.sum(y_value * y_value))
        self.assertAlmostEqual(report["scalar_lbfgs_last_sy"][1], expected_sy)
        self.assertAlmostEqual(report["scalar_lbfgs_last_yy"][1], expected_yy)
        self.assertAlmostEqual(
            report["scalar_lbfgs_raw_initial_inverse_scale"][1],
            expected_sy / expected_yy,
        )

    def test_direct_only_never_calls_original_selector(self):
        original = discrete.select_direction

        def forbidden_selector(*args, **kwargs):
            del args, kwargs
            raise AssertionError("selector must not run in direct-only mode")

        discrete.select_direction = forbidden_selector
        try:
            report = run_shared("direct_lbfgs_only")[8]
            self.assertTrue(np.all(report["direction_selection"] == "LBFGS"))
            with self.assertRaisesRegex(AssertionError, "selector must not run"):
                run_shared("scalar_selector")
        finally:
            discrete.select_direction = original

    def test_shared_solver_rejects_nonregistered_controls_and_bank(self):
        for updates in (
            {"inner_max_steps": 9},
            {"stop_mode": "r08"},
            {"relative_residual": 0.7},
            {"history_size": 6},
            {"kkt_atol": 1.0e-8},
        ):
            with self.assertRaises(ValueError):
                run_shared("direct_lbfgs_only", **updates)
        with self.assertRaisesRegex(
            scalar.ScalarLBFGSSafetyError, "preconditioner_bank=None"
        ):
            run_shared("direct_lbfgs_only", preconditioner_bank=object())

    def test_runtime_injects_aligned_audit_and_bank_is_zero_nf_unused_inside(self):
        installed = scalar.install_scalar_lbfgs_shim(
            discrete, "direct_lbfgs_only"
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "result.npz"

                def save_npz(path, **arrays):
                    np.savez(path, **arrays)

                fake_runner = SimpleNamespace(
                    DISCRETE_B040_IMPLEMENTATION_ID=(
                        installed.base_implementation_id
                    ),
                    atomic_save_npz=save_npz,
                    solve_frozen_discrete_b040_band=installed.original_solve,
                    ForceFieldPreconditionerBank=FakeBank,
                )
                installed.attach_runner(fake_runner)
                backend = SimpleNamespace(raw_joint_evaluations=0)
                bank = fake_runner.ForceFieldPreconditionerBank(backend)
                path, energies, forces, tangents, evaluators = one_image_problem()
                bank.build(path)
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
                    preconditioner_bank=bank,
                )
                report = result[8]
                fake_runner.atomic_save_npz(
                    destination,
                    frozen_subproblem=np.array("discrete_b040"),
                    discrete_implementation_id=np.array(
                        installed.implementation_id
                    ),
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
                    discrete_audit_selection=np.asarray(
                        report["direction_selection"], dtype="U8"
                    ),
                )
                with np.load(destination, allow_pickle=False) as saved:
                    self.assertEqual(
                        saved["discrete_scalar_lbfgs_mode"].item(),
                        "direct_lbfgs_only",
                    )
                    self.assertTrue(
                        bool(
                            saved[
                                "discrete_scalar_lbfgs_fixed_solver_preconditioner_is_none"
                            ]
                        )
                    )
                    self.assertFalse(
                        bool(
                            saved[
                                "discrete_scalar_lbfgs_ffrbk_applied_inside_fixed_solver"
                            ]
                        )
                    )
                    self.assertEqual(
                        int(
                            saved[
                                "discrete_scalar_lbfgs_ffrbk_solve_normal_calls_inside_fixed_solver"
                            ]
                        ),
                        0,
                    )
                    self.assertEqual(
                        int(saved["discrete_scalar_lbfgs_bank_build_count"]), 1
                    )
                    np.testing.assert_array_equal(
                        saved[
                            "discrete_scalar_lbfgs_bank_build_raw_joint_evaluation_delta"
                        ],
                        [0],
                    )
                    self.assertTrue(
                        bool(saved["discrete_scalar_lbfgs_bank_build_zero_nf"])
                    )
                    self.assertTrue(
                        np.all(
                            saved[
                                "discrete_scalar_lbfgs_audit_selection"
                            ]
                            == "LBFGS"
                        )
                    )
                    self.assertFalse(
                        np.any(
                            saved[
                                "discrete_scalar_lbfgs_audit_h0_constructed"
                            ]
                        )
                    )
                    self.assertFalse(
                        bool(
                            saved[
                                "discrete_scalar_lbfgs_absolute_kkt_stop_enabled"
                            ]
                        )
                    )
        finally:
            installed.restore()

    def test_bank_solve_inside_fixed_solver_fails_closed_and_restore_is_exact(self):
        installed = scalar.install_scalar_lbfgs_shim(
            discrete, "scalar_selector"
        )
        fake_runner = SimpleNamespace(
            DISCRETE_B040_IMPLEMENTATION_ID=installed.base_implementation_id,
            atomic_save_npz=lambda path, **arrays: None,
            solve_frozen_discrete_b040_band=installed.original_solve,
            ForceFieldPreconditionerBank=FakeBank,
        )
        original_build = FakeBank.build
        original_solve = FakeBank.solve_normal
        try:
            installed.attach_runner(fake_runner)
            bank = fake_runner.ForceFieldPreconditionerBank(
                SimpleNamespace(raw_joint_evaluations=0)
            )
            installed.audit.fixed_solver_depth = 1
            with self.assertRaisesRegex(
                scalar.ScalarLBFGSSafetyError, "forbidden inside"
            ):
                bank.solve_normal(0, np.ones(3), np.ones(3))
            self.assertEqual(
                installed.audit.bank_solve_normal_inside_fixed_solver, 1
            )
            installed.audit.fixed_solver_depth = 0
        finally:
            installed.restore()
        self.assertIs(FakeBank.build, original_build)
        self.assertIs(FakeBank.solve_normal, original_solve)

    def test_fresh_import_order_restores_runner_to_authoritative_base(self):
        installed = scalar.install_scalar_lbfgs_shim(
            discrete, "direct_lbfgs_only"
        )
        # This is the exact main-order invariant: a runner imported only
        # after frozen installation captures the shim pair immediately.
        fresh_runner = SimpleNamespace(
            DISCRETE_B040_IMPLEMENTATION_ID=discrete.IMPLEMENTATION_ID,
            solve_frozen_discrete_b040_band=(
                discrete.solve_frozen_discrete_b040_band
            ),
            atomic_save_npz=lambda path, **arrays: None,
            ForceFieldPreconditionerBank=FakeBank,
        )
        self.assertEqual(
            fresh_runner.DISCRETE_B040_IMPLEMENTATION_ID,
            installed.implementation_id,
        )
        self.assertIs(
            fresh_runner.solve_frozen_discrete_b040_band,
            discrete.solve_frozen_discrete_b040_band,
        )
        try:
            installed.attach_runner(fresh_runner)
        finally:
            installed.restore()
        self.assertEqual(
            fresh_runner.DISCRETE_B040_IMPLEMENTATION_ID,
            BASE_IMPLEMENTATION_ID,
        )
        self.assertIs(
            fresh_runner.solve_frozen_discrete_b040_band,
            installed.original_solve,
        )
        self.assertIs(
            discrete.solve_frozen_discrete_b040_band,
            installed.original_solve,
        )


if __name__ == "__main__":
    unittest.main()
