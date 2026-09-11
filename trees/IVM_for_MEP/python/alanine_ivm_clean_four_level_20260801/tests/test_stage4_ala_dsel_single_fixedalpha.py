from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np

import run_stage4_ala_discrete_single_fixedalpha as single
import run_stage4_ala_discrete_single_fixedalpha_grid as grid


class SingleFixedAlphaCommandTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("/tmp/ala-dsel-single-command-audit")

    def command(self, role: str):
        return single.single_run_command(
            self.root, 20.0, 254, "r1", role
        )

    def test_candidate_has_one_fixed_alpha_and_one_canonical_stop(self):
        command, label = self.command("candidate")
        options = single.command_options(command)
        self.assertEqual(options["--initial-path-file"], single.COMMON_INITIAL)
        self.assertEqual(options["--alpha"], "20")
        self.assertEqual(options["--alpha-mode"], "fixed")
        self.assertEqual(options["--frozen-subproblem"], "discrete_b040")
        self.assertEqual(options["--discrete-stop-mode"], "r08c10")
        self.assertEqual(options["--discrete-inner-max-steps"], "10")
        self.assertEqual(options["--discrete-residual-ratio"], "0.8")
        self.assertNotIn("--ala-dsel-trigger-iperp", options)
        self.assertEqual(options["--reparam-mode"], "state_curve")
        self.assertEqual(
            options["--state-curve-direct-objective-acceptance"],
            "per_image",
        )
        self.assertNotIn("--ala-dsel-alpha", options)
        self.assertNotIn("--alpha-schedule", options)
        self.assertNotIn("phase", label.lower())
        self.assertNotIn("continuation", label.lower())

    def test_matched_control_is_legacy_with_no_dsel_options(self):
        command, _ = self.command("matched_control")
        options = single.command_options(command)
        self.assertEqual(options["--frozen-subproblem"], "legacy")
        for flag in (
            "--discrete-inner-max-steps",
            "--discrete-stop-mode",
            "--discrete-residual-ratio",
            "--discrete-history",
            "--ala-dsel-trigger-iperp",
        ):
            self.assertNotIn(flag, options)
        self.assertEqual(options["--alpha"], "20")
        self.assertEqual(options["--alpha-mode"], "fixed")
        self.assertEqual(options["--reparam-mode"], "state_curve")

    def test_candidate_and_control_share_registered_outer_policy(self):
        candidate, _ = self.command("candidate")
        control, _ = self.command("matched_control")
        single.assert_matched_pair_commands(candidate, control, 20.0)
        candidate_options = single.command_options(candidate)
        control_options = single.command_options(control)
        routing = {"--run-label"}
        dsel = {
            "--frozen-subproblem",
            "--discrete-inner-max-steps",
            "--discrete-stop-mode",
            "--discrete-residual-ratio",
            "--discrete-history",
            "--ala-dsel-trigger-iperp",
        }
        for flag in routing | dsel:
            candidate_options.pop(flag, None)
            control_options.pop(flag, None)
        self.assertEqual(candidate_options, control_options)

        control[control.index("--reparam-cv-trigger") + 1] = "0.002"
        with self.assertRaisesRegex(RuntimeError, "command drift"):
            single.assert_matched_pair_commands(candidate, control, 20.0)

    def test_policy_audit_rejects_stop_or_reparam_drift(self):
        command, _ = self.command("candidate")
        command[command.index("--discrete-stop-mode") + 1] = "c10"
        with self.assertRaisesRegex(RuntimeError, "R08C10"):
            single.assert_single_run_policy(command, 20.0, "candidate")

        command, _ = self.command("candidate")
        command[command.index("--reparam-mode") + 1] = "state_curve_projected"
        with self.assertRaisesRegex(RuntimeError, "state_curve"):
            single.assert_single_run_policy(command, 20.0, "candidate")

    def test_formal_grid_is_canonical_and_has_no_stop_dimension(self):
        self.assertEqual(
            grid.CANONICAL_ALPHAS,
            (0.01, 0.1, 1.0, 10.0, 20.0, 30.0, 100.0, 1000.0, 10000.0, math.inf),
        )
        self.assertEqual(grid.CANONICAL_BUDGET, 254)
        self.assertEqual(grid.MEMBER_ROLES, ("candidate", "matched_control"))
        self.assertEqual(
            single.POLICY_VERSION,
            "stage4_ala_discrete_single_fixedalpha_v2",
        )
        self.assertEqual(
            grid.REPORTING_ANCHORS,
            (1.0, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4, 1.5e-4),
        )

    def test_single_run_complete_states_are_initial_plus_committed(self):
        saved = {
            "total_internal_joint_evaluations": np.asarray([20, 40, 60, 80]),
            "reparam_committed": np.asarray([False, False, True, False]),
        }
        self.assertTrue(
            np.array_equal(
                single.complete_candidate_calls(saved),
                np.asarray([20, 60]),
            )
        )


if __name__ == "__main__":
    unittest.main()
