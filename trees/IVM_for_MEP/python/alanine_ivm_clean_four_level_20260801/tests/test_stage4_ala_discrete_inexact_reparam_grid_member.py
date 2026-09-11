from __future__ import annotations

import itertools
from pathlib import Path
import unittest

import run_ivm_v9_discrete_ffrbk_takeover_grid_shim as runtime
import run_ivm_v9_total_internal_nf_budget as total_budget
import run_stage4_ala_discrete_inexact_reparam_grid_member as member


class InexactReparamGridMemberTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def test_registered_cartesian_grid_has_240_unique_cells(self):
        labels = set()
        hashes = set()
        for alpha, ratio, cap in itertools.product(
            member.ALPHAS, member.RESIDUAL_RATIOS, member.INNER_MAX_STEPS
        ):
            command, label = member.single_run_command(
                self.root, alpha, ratio, cap
            )
            labels.add(label)
            hashes.add(member.compact_command_sha256(command))
            self.assertEqual(
                command[-2:],
                [total_budget.TOTAL_INTERNAL_NF_FLAG, "5000"],
            )
        self.assertEqual(len(labels), 240)
        self.assertEqual(len(hashes), 240)

    def test_representative_command_retains_current_algorithm_identity(self):
        command, label = member.single_run_command(self.root, 1.0e6, 0.1, 50)
        options = member._options(command)
        self.assertEqual(Path(command[1]).name,
                         "run_ivm_v9_discrete_ffrbk_takeover_grid_shim.py")
        self.assertEqual(options["--alpha"], "1e+06")
        self.assertEqual(options["--discrete-residual-ratio"], "0.1")
        self.assertEqual(options["--discrete-inner-max-steps"], "50")
        self.assertEqual(options["--max-step"], "1")
        self.assertEqual(options["--discrete-step-multiplier-eta"], "1.1")
        self.assertEqual(options["--plbfgs-kkt-atol"], "0")
        self.assertEqual(options["--reparam-mode"], "state_curve")
        self.assertEqual(options["--reparam-every"], "1")
        self.assertEqual(options["--discrete-ffrbk-selector-mode"],
                         "one_way_takeover")
        self.assertEqual(options["--budget"], "5000")
        self.assertEqual(options[total_budget.TOTAL_INTERNAL_NF_FLAG], "5000")
        self.assertIn("a1e6_rr10_c50_tnf5000", label)

        controls, mode, eta, runner_argv = runtime.extract_runtime_arguments(
            command
        )
        with runtime.configured_selector_policy(
            runtime.fixed_selector, controls
        ) as selector:
            runtime.validate_inexact_reparam_runner_argv(
                selector, runner_argv, controls, mode, eta
            )

    def test_maxstep_means_inner_cap_not_physical_displacement(self):
        for cap in member.INNER_MAX_STEPS:
            command, _ = member.single_run_command(self.root, 1.0, 0.8, cap)
            options = member._options(command)
            self.assertEqual(options["--discrete-inner-max-steps"], str(cap))
            self.assertEqual(options["--max-step"], "1")

    def test_frozen_runtime_sources_remain_at_registered_hashes(self):
        expected = {
            "src/run_ivm_v9_spring_projected_production_20260730.py": (
                "9abd6e96bb9e6cead380a28c849fd11c15c5fe9a8854d77fe41a991240b0510f"
            ),
            "src/frozen_discrete_b040.py": (
                "8384a84666af463991df2e8d8a344779ae675feaddc73c12bcc82ff6ba324e22"
            ),
            "src/run_ivm_v9_discrete_selected_eta_shim.py": (
                "a2f071ea787a008ed001fd806b8952f84d8db90335b544308478f3f6ae2facf9"
            ),
            "src/run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py": (
                "62c6cf784795984c8104b4ba28fe2b0e15d657593648734d8f4a36ae03edfffc"
            ),
        }
        for relative, sha256 in expected.items():
            self.assertEqual(member.file_sha256(self.root / relative), sha256)

    def test_unregistered_controls_fail_closed(self):
        for args in ((15.0, 0.8, 10), (1.0, 0.9, 10), (1.0, 0.8, 250)):
            with self.assertRaises(ValueError):
                member.single_run_command(self.root, *args)


if __name__ == "__main__":
    unittest.main()

