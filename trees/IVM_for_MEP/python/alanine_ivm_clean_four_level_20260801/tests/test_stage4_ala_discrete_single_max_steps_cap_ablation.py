from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import run_stage4_ala_discrete_single_max_steps_cap_ablation as member
import run_stage4_ala_discrete_single_max_steps_cap_ablation_grid as grid


class MaxEvaluatedStepsCapAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def test_grid_tags_and_primary_eligibility_are_exact(self):
        self.assertEqual(member.NEW_CAPS, (1, 2, 3))
        self.assertEqual(member.REFERENCE_CAP, 10)
        self.assertEqual(member.PRIMARY_LOCK_ELIGIBLE_CAPS, (2, 3, 10))
        self.assertEqual(member.CANONICAL_ALPHA, 12.5)
        self.assertEqual(member.CANONICAL_RATIO, 0.8)
        self.assertEqual(member.CANONICAL_BUDGET, 254)
        self.assertEqual(member.LBFGS_HISTORY, 7)
        self.assertEqual([member.cap_tag(cap) for cap in member.NEW_CAPS], ["c01", "c02", "c03"])
        with self.assertRaisesRegex(ValueError, "Evaluated-step cap"):
            member.single_run_command(self.root, 10)

    def test_commands_change_only_routing_and_inner_cap(self):
        canonical = member.canonical_reference_command(self.root)
        canonical_options = member.coarse_single.command_options(canonical)
        for cap in member.NEW_CAPS:
            command, label = member.single_run_command(self.root, cap)
            member.assert_cap_only_scientific_change(command, canonical, cap)
            options = member.coarse_single.command_options(command)
            self.assertEqual(options["--alpha"], "12.5")
            self.assertEqual(options["--discrete-residual-ratio"], "0.8")
            self.assertEqual(options["--discrete-inner-max-steps"], str(cap))
            self.assertEqual(options["--discrete-stop-mode"], "r08c10")
            self.assertEqual(options["--discrete-history"], "7")
            self.assertEqual(options["--history"], "7")
            self.assertEqual(options["--budget"], "254")
            self.assertEqual(options["--reparam-mode"], "state_curve")
            self.assertEqual(
                options["--state-curve-direct-objective-acceptance"], "per_image"
            )
            self.assertIn(member.cap_tag(cap), label)
            for flag, expected in canonical_options.items():
                if flag in member.ROUTING_OPTIONS | {member.SCIENTIFIC_ABLATION_OPTION}:
                    continue
                self.assertEqual(options[flag], expected, flag)

    def _synthetic_result(self, cap: int, lbfgs: bool) -> dict[str, np.ndarray]:
        size = member.CANONICAL_BUDGET
        if cap == 1:
            outer = np.asarray([0, 1])
            steps = np.asarray([0, 0])
            selections = np.asarray(["H0", "H0"])
        else:
            outer = np.asarray([0, 0])
            steps = np.asarray([0, 1])
            selections = np.asarray(["H0", "LBFGS" if lbfgs else "H0"])
        h0 = int(np.count_nonzero(selections == "H0"))
        lbfgs_count = int(np.count_nonzero(selections == "LBFGS"))
        return {
            "active_alpha": np.full(size, 12.5),
            "alpha": np.asarray(12.5),
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
            "discrete_audit_global_relative_residual": np.asarray([0.9]),
            "discrete_audit_history_count": np.arange(1, len(steps) + 1),
            "discrete_audit_outer": outer,
            "discrete_audit_selection": selections,
            "discrete_audit_step": steps,
            "discrete_audit_stop_reason": np.asarray(["evaluated-inner-max-steps"]),
            "discrete_history": np.asarray(7),
            "discrete_h0_selection_count": np.asarray(h0),
            "discrete_implementation_id": np.asarray(member.coarse_single.EXPECTED_IMPLEMENTATION_ID),
            "discrete_inner_max_steps": np.asarray(cap),
            "discrete_lbfgs_selection_count": np.asarray(lbfgs_count),
            "discrete_residual_ratio": np.asarray(0.8),
            "discrete_stop_mode": np.asarray("r08c10"),
            "frozen_subproblem": np.asarray("discrete_b040"),
            # Non-finite embargoed data proves the validator never indexes it.
            "barrier_kcal": np.full(size, np.nan),
        }

    def test_cap1_is_h0_only_and_cap2_allows_lbfgs_takeover(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            cap1 = temporary / "cap1.npz"
            cap2 = temporary / "cap2.npz"
            np.savez(cap1, **self._synthetic_result(1, False))
            np.savez(cap2, **self._synthetic_result(2, True))
            one = member._validate_result_iperp_only(cap1, 1)
            two = member._validate_result_iperp_only(cap2, 2)
        self.assertEqual(one["lbfgs_selection_count"], 0)
        self.assertFalse(one["lbfgs_takeover_observed"])
        self.assertEqual(two["lbfgs_selection_count"], 1)
        self.assertTrue(two["lbfgs_takeover_observed"])
        self.assertNotIn("barrier_kcal", one["result_fields_accessed"])

    def test_cap1_validator_rejects_impossible_lbfgs_takeover(self):
        payload = self._synthetic_result(1, False)
        payload["discrete_audit_selection"] = np.asarray(["LBFGS", "H0"])
        payload["discrete_h0_selection_count"] = np.asarray(1)
        payload["discrete_lbfgs_selection_count"] = np.asarray(1)
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "invalid.npz"
            np.savez(result, **payload)
            with self.assertRaisesRegex(RuntimeError, "H0-only"):
                member._validate_result_iperp_only(result, 1)

    def test_result_access_sets_exclude_barrier(self):
        for keys in (member.COMMON_RESULT_KEYS, member.DISCRETE_RESULT_KEYS):
            self.assertFalse(any("barrier" in key.lower() for key in keys), keys)

    def test_preregistration_locks_ratio_chain_reuse_and_mechanism(self):
        planned = self.root / "reports" / (
            f"{grid.PROTOCOL_ID}_N254_r1_unit_test_never_written.json"
        )
        with mock.patch.object(grid, "_assert_no_outputs"):
            protocol = grid.build_preregistration(self.root, planned)
        self.assertEqual(protocol["new_max_evaluated_steps_caps"], [1, 2, 3])
        self.assertEqual(protocol["display_caps"], [1, 2, 3, 10])
        self.assertEqual(protocol["new_member_tags"], ["c01", "c02", "c03"])
        self.assertEqual(protocol["primary_lock_eligible_caps"], [2, 3, 10])
        self.assertTrue(protocol["cap1_role"]["mechanism_only"])
        self.assertFalse(protocol["cap1_role"]["primary_winner_eligible"])
        self.assertIn("at least one observed LBFGS", protocol["primary_success_criterion"])
        self.assertEqual(
            protocol["formal_ratio_chain"]["formal_ratio_analyzer_output"]["sha256"],
            grid.FORMAL_RATIO_ANALYSIS_SHA256,
        )
        self.assertEqual(
            protocol["canonical_common_initial_path"]["sha256"],
            grid.COMMON_INITIAL_FILE_SHA256,
        )
        self.assertEqual(
            protocol["canonical_common_initial_path"]["initial_path_array_sha256"],
            grid.COMMON_INITIAL_ARRAY_SHA256,
        )
        self.assertEqual(
            protocol["reference_mechanism_evidence"]["per_solve_evaluated_step_count_distribution"],
            {"1": 4, "3": 12, "4": 3, "6": 1},
        )
        self.assertEqual(protocol["reference_mechanism_evidence"]["maximum_observed_evaluated_steps"], 6)
        self.assertFalse(protocol["no_primary_dominator_rule"]["winner_claim_allowed"])
        self.assertFalse(protocol["no_primary_dominator_rule"]["post_hoc_grid_expansion_allowed"])
        self.assertFalse(protocol["no_primary_dominator_rule"]["barrier_unembargo_allowed"])
        with mock.patch.object(grid, "_assert_no_outputs"):
            grid.validate_preregistration(protocol, self.root)
            drifted = copy.deepcopy(protocol)
            drifted["formal_ratio_chain"]["formal_ratio_analyzer_output"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "Formal ratio chain drifted"):
                grid.validate_preregistration(drifted, self.root)

    def test_paths_no_overwrite_and_cli_separation(self):
        paths = [member.member_paths(self.root, cap) for cap in member.NEW_CAPS]
        self.assertEqual(len({str(path["label"]) for path in paths}), 3)
        for path in paths:
            self.assertIn(member.MEMBER_POLICY_VERSION, str(path["result_dir"]))
        parser = grid.argument_parser()
        prereg = parser.parse_args(["--root", str(self.root), "--preregister-only"])
        launch = parser.parse_args([
            "--root", str(self.root), "--preregistration", str(self.root / "reports/x.json")
        ])
        self.assertTrue(prereg.preregister_only)
        self.assertIsNone(prereg.preregistration)
        self.assertFalse(launch.preregister_only)
        self.assertIsNotNone(launch.preregistration)


if __name__ == "__main__":
    unittest.main()
