from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

import run_stage4_ala_discrete_single_policy_pilot as pilot
import run_stage4_ala_dsel_single_fixedalpha as single


class DiscreteSinglePolicyPilotTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("/tmp/ala-discrete-policy-pilot")

    def test_policy_triples_change_only_registered_solver_package(self):
        for policy in pilot.POLICIES:
            commands = {
                role: pilot.pilot_command(
                    self.root, policy, role, "r1"
                )[0]
                for role in pilot.SOLVER_ROLES
            }
            pilot.assert_policy_triple(commands, policy, pilot.PILOT_BUDGET)

    def test_primary_is_direct_discrete_r08c10(self):
        command, _ = pilot.pilot_command(
            self.root, "precision_state_curve", "discrete_b040", "r1"
        )
        options = single.command_options(command)
        self.assertEqual(options["--frozen-subproblem"], "discrete_b040")
        self.assertEqual(options["--discrete-stop-mode"], "r08c10")
        self.assertEqual(options["--discrete-inner-max-steps"], "10")
        self.assertEqual(options["--discrete-residual-ratio"], "0.8")
        self.assertEqual(options["--alpha"], "20")
        self.assertEqual(options["--alpha-mode"], "fixed")

    def test_policy_bundles_are_exactly_distinct(self):
        precision, _ = pilot.pilot_command(
            self.root, "precision_state_curve", "legacy", "r1"
        )
        projected, _ = pilot.pilot_command(
            self.root, "projected_pathsum", "legacy", "r1"
        )
        precision_options = single.command_options(precision)
        projected_options = single.command_options(projected)
        self.assertEqual(precision_options["--reparam-mode"], "state_curve")
        self.assertEqual(
            precision_options["--state-curve-direct-objective-acceptance"],
            "per_image",
        )
        self.assertEqual(
            projected_options["--reparam-mode"], "state_curve_projected"
        )
        self.assertEqual(
            projected_options["--state-curve-direct-objective-acceptance"],
            "path_sum",
        )

    def test_complete_state_definition_depends_on_policy(self):
        committed = np.asarray([False, False, True, False])
        self.assertTrue(
            np.array_equal(
                pilot.complete_state_mask("precision_state_curve", committed),
                np.asarray([True, False, True, False]),
            )
        )
        self.assertTrue(
            np.array_equal(
                pilot.complete_state_mask("projected_pathsum", committed),
                np.ones(4, dtype=bool),
            )
        )


if __name__ == "__main__":
    unittest.main()
