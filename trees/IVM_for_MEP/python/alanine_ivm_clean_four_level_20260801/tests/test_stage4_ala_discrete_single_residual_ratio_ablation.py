from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import run_stage4_ala_discrete_single_residual_ratio_ablation as member
import run_stage4_ala_discrete_single_residual_ratio_ablation_grid as grid


class ResidualRatioAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def test_grid_is_exactly_two_new_ratios(self):
        self.assertEqual(member.NEW_RATIOS, (0.9, 0.5))
        self.assertEqual(member.REFERENCE_RATIO, 0.8)
        self.assertEqual(member.CANONICAL_ALPHA, 12.5)
        self.assertEqual(member.CANONICAL_BUDGET, 254)
        self.assertEqual(member.EVALUATED_STEP_CAP, 10)
        self.assertEqual(member.LBFGS_HISTORY, 7)
        self.assertEqual(member.ratio_tag(0.9), "rr090c10")
        self.assertEqual(member.ratio_tag(0.5), "rr050c10")
        with self.assertRaisesRegex(ValueError, "Residual ratio"):
            member.single_run_command(self.root, 0.8)

    def test_each_command_changes_only_routing_and_ratio(self):
        canonical = member.canonical_reference_command(self.root)
        canonical_options = member.coarse_single.command_options(canonical)
        for ratio in member.NEW_RATIOS:
            command, label = member.single_run_command(self.root, ratio)
            member.assert_ratio_only_scientific_change(
                command, canonical, ratio
            )
            options = member.coarse_single.command_options(command)
            self.assertEqual(options["--alpha"], "12.5")
            self.assertEqual(options["--budget"], "254")
            self.assertEqual(options["--discrete-inner-max-steps"], "10")
            self.assertEqual(options["--discrete-history"], "7")
            self.assertEqual(options["--history"], "7")
            self.assertEqual(options["--discrete-stop-mode"], "r08c10")
            self.assertEqual(float(options["--discrete-residual-ratio"]), ratio)
            self.assertEqual(options["--reparam-mode"], "state_curve")
            self.assertEqual(
                options["--state-curve-direct-objective-acceptance"],
                "per_image",
            )
            self.assertIn(member.ratio_tag(ratio), label)
            for flag, value in canonical_options.items():
                if flag in member.ROUTING_OPTIONS | {
                    member.SCIENTIFIC_ABLATION_OPTION
                }:
                    continue
                self.assertEqual(options[flag], value, flag)

    def test_paths_are_isolated_and_unambiguous(self):
        paths_09 = member.member_paths(self.root, 0.9)
        paths_05 = member.member_paths(self.root, 0.5)
        self.assertNotEqual(paths_09["label"], paths_05["label"])
        for ratio, paths in ((0.9, paths_09), (0.5, paths_05)):
            tag = member.ratio_tag(ratio)
            self.assertIn(member.MEMBER_POLICY_VERSION, str(paths["result_dir"]))
            self.assertIn(tag, str(paths["audit"]))
            self.assertIn(tag, str(paths["member_log"]))
            self.assertIn(tag, str(paths["launcher_log"]))
            self.assertNotIn("r08c10_a12p5", str(paths["label"]))

    def test_result_audit_is_barrier_blind(self):
        size = member.CANONICAL_BUDGET
        common = {
            "active_alpha": np.full(size, member.CANONICAL_ALPHA),
            "alpha": np.asarray(member.CANONICAL_ALPHA),
            "alpha_mode": np.asarray("fixed"),
            "configuration_sha256": np.asarray("a" * 64),
            "i_perp": np.linspace(1.0, 0.001, size),
            "initial_path_sha256": np.asarray("b" * 64),
            "reparam_committed": np.arange(size) % 2 == 0,
            "reparam_mode": np.asarray("state_curve"),
            "segment_length_cv": np.zeros(size),
            "segment_length_ratio": np.ones(size),
            "state_curve_direct_objective_acceptance": np.asarray("per_image"),
            "stop_complete_iperp": np.asarray(-1.0),
            "total_internal_joint_evaluations": np.arange(1, size + 1),
            "discrete_all_returned_states_evaluated": np.asarray(True),
            "discrete_audit_global_relative_residual": np.asarray([0.4]),
            "discrete_audit_history_count": np.asarray([1, 2]),
            "discrete_audit_selection": np.asarray(["H0", "LBFGS"]),
            "discrete_audit_stop_reason": np.asarray(
                ["evaluated-relative-natural-residual"]
            ),
            "discrete_history": np.asarray(7),
            "discrete_h0_selection_count": np.asarray(1),
            "discrete_implementation_id": np.asarray(
                member.coarse_single.EXPECTED_IMPLEMENTATION_ID
            ),
            "discrete_inner_max_steps": np.asarray(10),
            "discrete_lbfgs_selection_count": np.asarray(1),
            "discrete_residual_ratio": np.asarray(0.5),
            "discrete_stop_mode": np.asarray("r08c10"),
            "frozen_subproblem": np.asarray("discrete_b040"),
            # A non-finite embargoed field proves the validator never indexes it.
            "barrier_kcal": np.full(size, np.nan),
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "candidate.npz"
            np.savez(result, **common)
            summary = member._validate_result_iperp_only(result, 0.5)
        self.assertNotIn("barrier_kcal", summary["result_fields_accessed"])
        self.assertEqual(summary["h0_selection_count"], 1)
        self.assertEqual(summary["lbfgs_selection_count"], 1)
        self.assertEqual(summary["maximum_history_count"], 2)

    def test_all_result_access_sets_exclude_barrier(self):
        for keys in (member.COMMON_RESULT_KEYS, member.DISCRETE_RESULT_KEYS):
            self.assertFalse(any("barrier" in key.lower() for key in keys), keys)

    def test_preregistration_locks_decision_and_reuse_hashes(self):
        planned = self.root / "reports" / (
            f"{grid.PROTOCOL_ID}_N254_r1_unit_test_never_written.json"
        )
        # This content-contract test must remain repeatable after the registered
        # campaign has been launched.  Clean-start refusal is a launcher concern,
        # so isolate only that filesystem guard while retaining every hash check.
        with mock.patch.object(grid, "_assert_no_outputs"):
            protocol = grid.build_preregistration(self.root, planned)
        self.assertEqual(protocol["new_residual_ratios"], [0.9, 0.5])
        self.assertEqual(protocol["display_grid"], [0.9, 0.8, 0.5])
        self.assertEqual(protocol["fixed_alpha"], 12.5)
        self.assertEqual(protocol["new_member_tags"], ["rr090c10", "rr050c10"])
        self.assertTrue(protocol["reference_ratio_candidate_reused_not_rerun"])
        self.assertTrue(protocol["same_alpha_legacy_control_reused_not_rerun"])
        self.assertFalse(protocol["barrier_fields_accessed_by_wrapper_or_launcher"])
        self.assertIn("tightest eligible I_perp", protocol["primary_success_criterion"])
        self.assertEqual(
            protocol["formal_refinement_analysis"]["sha256"],
            grid.FORMAL_REFINEMENT_ANALYSIS_SHA256,
        )
        self.assertEqual(
            protocol["canonical_common_initial_path"]["sha256"],
            grid.COMMON_INITIAL_FILE_SHA256,
        )
        self.assertEqual(
            protocol["canonical_common_initial_path"][
                "initial_path_array_sha256"
            ],
            grid.COMMON_INITIAL_ARRAY_SHA256,
        )
        self.assertEqual(
            set(protocol["reused_inputs"]),
            set(grid.EXPECTED_REUSE_HASHES),
        )
        self.assertFalse(
            protocol["no_primary_dominator_rule"]["winner_claim_allowed"]
        )
        self.assertFalse(
            protocol["no_primary_dominator_rule"]["barrier_unembargo_allowed"]
        )
        with mock.patch.object(grid, "_assert_no_outputs"):
            grid.validate_preregistration(protocol, self.root)

            drifted = copy.deepcopy(protocol)
            drifted["reused_inputs"]["reference_result"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "Hash drift"):
                grid.validate_preregistration(drifted, self.root)

    def test_preregister_and_launch_cli_are_separate(self):
        parser = grid.argument_parser()
        preregister = parser.parse_args(
            ["--root", str(self.root), "--preregister-only"]
        )
        self.assertTrue(preregister.preregister_only)
        self.assertIsNone(preregister.preregistration)
        launch = parser.parse_args(
            [
                "--root",
                str(self.root),
                "--preregistration",
                str(self.root / "reports/example.json"),
            ]
        )
        self.assertFalse(launch.preregister_only)
        self.assertIsNotNone(launch.preregistration)


if __name__ == "__main__":
    unittest.main()
