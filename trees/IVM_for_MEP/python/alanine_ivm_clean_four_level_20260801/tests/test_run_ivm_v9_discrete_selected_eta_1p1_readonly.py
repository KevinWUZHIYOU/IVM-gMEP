from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

import frozen_discrete_b040 as discrete
import run_ivm_v9_discrete_selected_eta_shim as eta_shim
import run_stage4_ala_discrete_single_eta1p1_alpha15_followup as eta_member
from tests.test_frozen_discrete_b040 import (
    IdentityH0,
    one_image_problem,
    quadratic,
)


ETA = 1.1


class SelectedEtaOnePointOneReadonlyTests(unittest.TestCase):
    def tearDown(self):
        self.assertEqual(
            discrete.IMPLEMENTATION_ID,
            "ala-frozen-discrete-ffrbk-dsel-m7-evaluated-v1",
        )

    def test_validate_and_binary64_implementation_id_are_exact(self):
        self.assertEqual(eta_shim.validate_eta(ETA), ETA)
        self.assertEqual(
            eta_shim.eta_token(ETA), "0x1d199999999999app0"
        )
        implementation = eta_shim.shim_implementation_id(
            discrete.IMPLEMENTATION_ID, ETA
        )
        self.assertEqual(
            implementation,
            "ala-frozen-discrete-ffrbk-dsel-m7-evaluated-v1+"
            "post-select-selected-step-eta-recap-v1+"
            "eta-0x1d199999999999app0",
        )
        self.assertNotEqual(
            implementation,
            eta_shim.shim_implementation_id(discrete.IMPLEMENTATION_ID, 1.0),
        )

    def test_alpha15_member_is_exact_formal_command_clone_plus_eta(self):
        root = Path(__file__).resolve().parents[1]
        formal_command_path = root / (
            "logs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/"
            "members/"
            "stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
            "r08c10_a15_N254_discrete_b040_r1/command.json"
        )
        self.assertEqual(
            hashlib.sha256(formal_command_path.read_bytes()).hexdigest(),
            "9bf0458e6d0f898ddaf76eaa9758531887c677853b4ce32b4bf98c3ea46125c8",
        )
        formal_command = json.loads(formal_command_path.read_text())
        self.assertEqual(
            eta_member.canonical_reference_command(root), formal_command
        )

        command, label = eta_member.single_run_command(root)
        expected = list(formal_command)
        expected[1] = str(
            (root / "src/run_ivm_v9_discrete_selected_eta_shim.py").resolve()
        )
        for flag, value in (
            (
                "--output-dir",
                f"runs/{eta_member.MEMBER_POLICY_VERSION}/members",
            ),
            ("--run-label", label),
        ):
            position = expected.index(flag)
            expected[position + 1] = value
        expected.extend([eta_shim.ETA_FLAG, "1.1"])
        self.assertEqual(command, expected)

        reference_options = {
            formal_command[index]: formal_command[index + 1]
            for index in range(len(formal_command) - 1)
            if formal_command[index].startswith("--")
            and formal_command[index + 1].startswith("--") is False
        }
        candidate_options = {
            command[index]: command[index + 1]
            for index in range(len(command) - 1)
            if command[index].startswith("--")
            and command[index + 1].startswith("--") is False
        }
        for flag, value in reference_options.items():
            if flag not in eta_member.ROUTING_OPTIONS:
                self.assertEqual(candidate_options[flag], value)
        self.assertEqual(candidate_options["--alpha"], "15")
        self.assertEqual(candidate_options[eta_shim.ETA_FLAG], "1.1")

    def test_post_original_selector_scaling_is_exact_without_cap(self):
        installed = eta_shim.install_frozen_eta_shim(discrete, ETA)
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
            np.testing.assert_allclose(
                selected["step"], ETA * lbfgs_base, rtol=0.0, atol=0.0
            )
            event = installed.audit.standalone[-1]
            self.assertEqual(event.requested_multiplier, ETA)
            self.assertAlmostEqual(event.applied_multiplier, ETA, places=15)
            self.assertFalse(event.cap_hit)
        finally:
            installed.restore()

    def test_eta_one_point_one_reuses_same_dynamic_recap(self):
        installed = eta_shim.install_frozen_eta_shim(discrete, ETA)
        try:
            gradient = np.asarray([(0.0, -1.0, 0.0)])
            tangents = np.asarray([(1.0, 0.0, 0.0)])
            h0 = discrete.screen_direction(
                "H0", [(0.0, 0.95, 0.0)], gradient, tangents, 1.0
            )
            lbfgs = discrete.screen_direction(
                "LBFGS", [(0.0, 0.5, 0.0)], gradient, tangents, 1.0
            )
            selected = discrete.select_direction(h0, lbfgs)
            self.assertEqual(selected["label"], "H0")
            np.testing.assert_allclose(
                selected["step"], [[0.0, 1.0, 0.0]], rtol=0.0, atol=1e-15
            )
            event = installed.audit.standalone[-1]
            self.assertTrue(event.cap_hit)
            self.assertAlmostEqual(
                event.requested_max_atomic_displacement, ETA * 0.95
            )
            self.assertAlmostEqual(event.applied_max_atomic_displacement, 1.0)
            self.assertAlmostEqual(event.applied_multiplier, 1.0 / 0.95)
        finally:
            installed.restore()

    def test_one_step_solver_charges_no_extra_physical_evaluation(self):
        def solve_once():
            path, energies, forces, tangents, evaluators = one_image_problem(
                quadratic(0.15)
            )
            result = discrete.solve_frozen_discrete_b040_band(
                path,
                energies,
                forces,
                tangents,
                evaluators,
                np.inf,
                10.0,
                1,
                inner_max_steps=1,
                stop_mode="c10",
                history_size=7,
                preconditioner_bank=IdentityH0(),
            )
            return result, evaluators

        reference, reference_evaluators = solve_once()
        installed = eta_shim.install_frozen_eta_shim(discrete, ETA)
        try:
            candidate, candidate_evaluators = solve_once()
            np.testing.assert_array_equal(reference[3], [1])
            np.testing.assert_array_equal(candidate[3], [1])
            self.assertEqual(reference[8]["joint_steps"], 1)
            self.assertEqual(candidate[8]["joint_steps"], 1)
            self.assertEqual(len(installed.audit.committed), 1)
            self.assertEqual(
                reference_evaluators[1].counters.joint_evaluations,
                candidate_evaluators[1].counters.joint_evaluations,
            )
            self.assertEqual(
                candidate_evaluators[1].counters.joint_evaluations, 2
            )
        finally:
            installed.restore()


if __name__ == "__main__":
    unittest.main()
