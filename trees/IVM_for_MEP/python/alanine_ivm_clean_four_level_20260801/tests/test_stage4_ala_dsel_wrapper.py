from __future__ import annotations

from pathlib import Path
import unittest

import run_stage4_ala_dsel_uniform as dsel


class AlaDselWrapperTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("/tmp/ala-dsel-command-audit")
        self.continuation = self.root / "continuation.npz"

    def commands(self):
        phase_a, label_a = dsel.phase_a_command(
            self.root, 20.0, 34, "r08", "r1", "A34_B220", 8e-4
        )
        phase_b, label_b = dsel.phase_b_command(
            self.root,
            20.0,
            220,
            self.continuation,
            "r08",
            "r1",
            -1.0,
            "A34_B220",
            8e-4,
        )
        return phase_a, phase_b, label_a, label_b

    def test_exact_bidirectional_outer_policy_audit(self):
        phase_a, phase_b, _, _ = self.commands()
        dsel.assert_frozen_outer_policy(
            self.root, 20.0, phase_a, phase_b, self.continuation
        )

    def test_unexpected_scientific_flag_is_rejected(self):
        phase_a, phase_b, _, _ = self.commands()
        phase_a += ["--unexpected-scientific-switch", "123"]
        with self.assertRaisesRegex(RuntimeError, "command drift"):
            dsel.assert_frozen_outer_policy(
                self.root, 20.0, phase_a, phase_b, self.continuation
            )

    def test_wrong_phase_a_initial_path_is_rejected(self):
        phase_a, phase_b, _, _ = self.commands()
        index = phase_a.index("--initial-path-file") + 1
        phase_a[index] = "inputs/wrong_path.npz"
        with self.assertRaisesRegex(RuntimeError, "command drift|initial path"):
            dsel.assert_frozen_outer_policy(
                self.root, 20.0, phase_a, phase_b, self.continuation
            )

    def test_execution_tag_names_both_phase_results(self):
        _, _, label_a, label_b = self.commands()
        self.assertIn("A34_B220", label_a)
        self.assertIn("A34_B220", label_b)
        self.assertIn("a20_t0p0008", label_a)
        self.assertIn("a20_t0p0008", label_b)

    def test_fixed_alpha_and_trigger_are_identical_in_both_phases(self):
        phase_a, phase_b, _, _ = self.commands()
        for command in (phase_a, phase_b):
            self.assertEqual(dsel.command_value(command, "--alpha"), "20")
            self.assertEqual(dsel.command_value(command, "--alpha-mode"), "fixed")
            self.assertEqual(
                dsel.command_value(command, "--ala-dsel-trigger-iperp"),
                "0.0008",
            )

    def test_alpha_switch_or_phase_specific_trigger_is_rejected(self):
        phase_a, phase_b, _, _ = self.commands()
        phase_b[phase_b.index("--alpha") + 1] = "40"
        with self.assertRaisesRegex(RuntimeError, "fixed alpha"):
            dsel.assert_frozen_outer_policy(
                self.root, 20.0, phase_a, phase_b, self.continuation
            )

        phase_a, phase_b, _, _ = self.commands()
        phase_b[phase_b.index("--ala-dsel-trigger-iperp") + 1] = "0.001"
        with self.assertRaisesRegex(RuntimeError, "different DSEL triggers"):
            dsel.assert_frozen_outer_policy(
                self.root, 20.0, phase_a, phase_b, self.continuation
            )


if __name__ == "__main__":
    unittest.main()
