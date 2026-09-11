from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

import frozen_discrete_b040 as discrete
import run_ivm_v9_discrete_selected_eta_shim as eta_shim


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
        self.counters.energy_requests += int(property_name in {"energy", "both"})
        self.counters.force_requests += int(property_name in {"force", "both"})
        self.counters.both_requests += int(property_name == "both")
        key = self._key(position)
        if key in self.cache:
            self.counters.cache_hits += 1
            energy, force = self.cache[key]
        else:
            energy, force = self.potential(np.asarray(position, dtype=np.float64))
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
    def solve_normal(self, image_index, vector, tangent, diagonal_shift=0.0):
        del image_index, tangent, diagonal_shift
        return np.asarray(vector, dtype=np.float64)


def quadratic(position):
    stiffness = 0.15
    energy = 0.5 * stiffness * float(position[1] ** 2)
    force = np.zeros_like(position)
    force[1] = -stiffness * position[1]
    return energy, force


def one_image_problem():
    path = np.asarray(
        [(-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)],
        dtype=np.float64,
    )
    evaluators = [MockEvaluator(quadratic) for _ in range(3)]
    energies, forces = discrete.evaluate_path(path, evaluators)
    tangents = np.asarray([(1.0, 0.0, 0.0)], dtype=np.float64)
    return path, energies, forces, tangents, evaluators


def run_solver():
    path, energies, forces, tangents, evaluators = one_image_problem()
    return discrete.solve_frozen_discrete_b040_band(
        path,
        energies,
        forces,
        tangents,
        evaluators,
        np.inf,
        10.0,
        10,
        inner_max_steps=10,
        stop_mode="r08c10",
        relative_residual=0.8,
        history_size=7,
        preconditioner_bank=IdentityH0(),
    )


class EtaShimTests(unittest.TestCase):
    def tearDown(self):
        # Each test restores explicitly; this additionally detects leakage.
        self.assertEqual(
            discrete.IMPLEMENTATION_ID,
            "chignolin-frozen-discrete-ffrbk-dsel-m7-evaluated-v1",
        )

    def test_argument_and_exact_id_validation(self):
        eta, cleaned = eta_shim.extract_eta_argument(
            ["shim.py", "--alpha", "12.5", eta_shim.ETA_FLAG, "1.25"]
        )
        self.assertEqual(eta, 1.25)
        self.assertEqual(cleaned, ["shim.py", "--alpha", "12.5"])
        self.assertNotEqual(
            eta_shim.shim_implementation_id("base", 1.25),
            eta_shim.shim_implementation_id("base", 1.5),
        )
        for argv in (
            ["shim.py"],
            ["shim.py", eta_shim.ETA_FLAG, "1", eta_shim.ETA_FLAG, "2"],
        ):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                eta_shim.extract_eta_argument(argv)
        for value in (0.0, -1.0, np.inf, np.nan):
            with self.assertRaises(ValueError):
                eta_shim.validate_eta(value)

    def test_original_selector_runs_before_only_selected_step_is_scaled(self):
        installed = eta_shim.install_frozen_eta_shim(discrete, 2.0)
        try:
            gradient = np.asarray(
                [(0.0, -1.0, 0.0, 0.0, -1.0, 0.0)]
            )
            tangents = np.asarray(
                [(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
            )
            h0 = discrete.screen_direction(
                "H0",
                [(0.0, 0.2, 0.0, 0.0, 0.0, 0.0)],
                gradient,
                tangents,
                1.0,
            )
            lbfgs = discrete.screen_direction(
                "LBFGS",
                [(0.0, 0.15, 0.0, 0.0, 0.15, 0.0)],
                gradient,
                tangents,
                1.0,
            )
            h0_base = h0["step"].copy()
            lbfgs_base = lbfgs["step"].copy()
            selected = discrete.select_direction(h0, lbfgs)

            self.assertEqual(selected["label"], "LBFGS")
            self.assertIsNot(selected, lbfgs)
            np.testing.assert_array_equal(h0["step"], h0_base)
            np.testing.assert_array_equal(lbfgs["step"], lbfgs_base)
            np.testing.assert_array_equal(selected["step"], 2.0 * lbfgs_base)
            event = installed.audit.standalone[-1]
            self.assertEqual(event.requested_multiplier, 2.0)
            self.assertEqual(event.applied_multiplier, 2.0)
            self.assertFalse(event.cap_hit)
        finally:
            installed.restore()

    def test_selected_step_is_recapped_by_same_dynamic_cap(self):
        installed = eta_shim.install_frozen_eta_shim(discrete, 2.0)
        try:
            gradient = np.asarray([(0.0, -1.0, 0.0)])
            tangents = np.asarray([(1.0, 0.0, 0.0)])
            h0 = discrete.screen_direction(
                "H0", [(0.0, 0.6, 0.0)], gradient, tangents, 1.0
            )
            lbfgs = discrete.screen_direction(
                "LBFGS", [(0.0, 0.5, 0.0)], gradient, tangents, 1.0
            )
            selected = discrete.select_direction(h0, lbfgs)
            self.assertEqual(selected["label"], "H0")
            self.assertAlmostEqual(
                np.max(np.linalg.norm(selected["step"].reshape(-1, 3), axis=1)),
                1.0,
            )
            event = installed.audit.standalone[-1]
            self.assertTrue(event.cap_hit)
            self.assertAlmostEqual(event.requested_max_atomic_displacement, 1.2)
            self.assertAlmostEqual(event.applied_max_atomic_displacement, 1.0)
            self.assertAlmostEqual(event.applied_multiplier, 1.0 / 0.6)
        finally:
            installed.restore()

    def test_eta_one_returns_original_selected_object(self):
        installed = eta_shim.install_frozen_eta_shim(discrete, 1.0)
        try:
            gradient = np.asarray([(0.0, -1.0, 0.0)])
            tangents = np.asarray([(1.0, 0.0, 0.0)])
            h0 = discrete.screen_direction(
                "H0", [(0.0, 0.2, 0.0)], gradient, tangents, 1.0
            )
            lbfgs = discrete.screen_direction(
                "LBFGS", [(0.0, 0.1, 0.0)], gradient, tangents, 1.0
            )
            self.assertIs(discrete.select_direction(h0, lbfgs), h0)
            event = installed.audit.standalone[-1]
            self.assertEqual(event.applied_multiplier, 1.0)
            self.assertFalse(event.cap_hit)
        finally:
            installed.restore()

    def test_eta_one_solver_output_is_exact_parity(self):
        reference = run_solver()
        installed = eta_shim.install_frozen_eta_shim(discrete, 1.0)
        try:
            candidate = run_solver()
            eta_shim.assert_eta_one_exact_parity(reference, candidate)
            self.assertEqual(
                len(installed.audit.committed),
                int(candidate[8]["joint_steps"]),
            )
        finally:
            installed.restore()

    def test_runtime_save_injects_aligned_committed_audit(self):
        installed = eta_shim.install_frozen_eta_shim(discrete, 1.5)
        try:
            result = run_solver()
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
                )
                installed.attach_runner(fake_runner)
                self.assertIs(
                    fake_runner.solve_frozen_discrete_b040_band,
                    discrete.solve_frozen_discrete_b040_band,
                )
                fake_runner.atomic_save_npz(
                    destination,
                    frozen_subproblem=np.array("discrete_b040"),
                    discrete_implementation_id=np.array(
                        installed.implementation_id
                    ),
                    discrete_audit_selection=np.asarray(
                        result[8]["direction_selection"], dtype="U8"
                    ),
                )
                with np.load(destination, allow_pickle=False) as saved:
                    self.assertEqual(
                        float(saved["discrete_step_multiplier_eta"]), 1.5
                    )
                    np.testing.assert_array_equal(
                        saved["discrete_step_multiplier_audit_selection"],
                        result[8]["direction_selection"],
                    )
                    requested = saved[
                        "discrete_step_multiplier_audit_requested"
                    ]
                    applied = saved["discrete_step_multiplier_audit_applied"]
                    self.assertEqual(requested.shape, applied.shape)
                    self.assertEqual(requested.size, result[8]["joint_steps"])
                    np.testing.assert_array_equal(requested, 1.5)
                    self.assertTrue(
                        bool(saved["discrete_step_multiplier_selector_isolated"])
                    )
                    self.assertTrue(
                        bool(
                            saved[
                                "discrete_step_multiplier_dynamic_cap_reapplied"
                            ]
                        )
                    )
                installed.restore()
                self.assertIs(
                    fake_runner.solve_frozen_discrete_b040_band,
                    installed.original_solve,
                )
        finally:
            installed.restore()


if __name__ == "__main__":
    unittest.main()
