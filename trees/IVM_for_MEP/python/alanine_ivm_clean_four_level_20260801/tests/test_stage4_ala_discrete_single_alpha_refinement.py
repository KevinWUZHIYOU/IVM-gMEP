from __future__ import annotations

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import run_stage4_ala_discrete_single_alpha_refinement as member
import run_stage4_ala_discrete_single_alpha_refinement_grid as grid
import run_stage4_ala_discrete_single_fixedalpha as coarse


class DiscreteSingleAlphaRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def test_grid_and_neighborhood_are_exactly_preregistered(self):
        self.assertEqual(member.REFINEMENT_ALPHAS, (5.0, 7.5, 12.5, 15.0))
        self.assertEqual(grid.REFINEMENT_ALPHAS, member.REFINEMENT_ALPHAS)
        self.assertEqual(grid.COARSE_BRACKET, (1.0, 20.0))
        self.assertEqual(grid.LOCKED_CENTER_ALPHA, 10.0)
        self.assertEqual(grid.UPPER_BOUNDARY_ALPHA, 20.0)
        self.assertEqual(member.CANONICAL_BUDGET, 254)
        self.assertEqual(member.CANONICAL_REPLICATE, "r1")
        self.assertNotIn(10.0, member.REFINEMENT_ALPHAS)
        self.assertNotIn(20.0, member.REFINEMENT_ALPHAS)
        for alpha in member.REFINEMENT_ALPHAS:
            self.assertGreater(alpha, 1.0)
            self.assertGreaterEqual(alpha, 5.0)
            self.assertLess(alpha, 20.0)

    def test_every_member_changes_only_immutable_routing(self):
        for alpha in member.REFINEMENT_ALPHAS:
            commands = {}
            for role in member.MEMBER_ROLES:
                command, label = member.single_run_command(
                    self.root, alpha, 254, "r1", role
                )
                canonical, _ = coarse.single_run_command(
                    self.root, alpha, 254, "r1", role
                )
                member.assert_routing_only_change(command, canonical, alpha, role)
                options = coarse.command_options(command)
                self.assertEqual(options["--alpha"], f"{alpha:g}")
                self.assertEqual(options["--alpha-mode"], "fixed")
                self.assertEqual(options["--reparam-mode"], "state_curve")
                self.assertEqual(
                    options["--state-curve-direct-objective-acceptance"],
                    "per_image",
                )
                self.assertEqual(options["--budget"], "254")
                self.assertEqual(options["--run-label"], label)
                self.assertIn(member.MEMBER_POLICY_VERSION, label)
                commands[role] = command
            member.assert_matched_pair_commands(
                commands["candidate"], commands["matched_control"], alpha
            )
            candidate = coarse.command_options(commands["candidate"])
            control = coarse.command_options(commands["matched_control"])
            self.assertEqual(candidate["--frozen-subproblem"], "discrete_b040")
            self.assertEqual(candidate["--discrete-stop-mode"], "r08c10")
            self.assertEqual(candidate["--discrete-inner-max-steps"], "10")
            self.assertEqual(candidate["--discrete-residual-ratio"], "0.8")
            self.assertEqual(control["--frozen-subproblem"], "legacy")
            self.assertNotIn("--discrete-stop-mode", control)

    def test_refinement_paths_are_isolated_from_frozen_coarse_paths(self):
        paths = member.member_paths(self.root, 5.0, "candidate")
        command, _ = member.single_run_command(
            self.root, 5.0, 254, "r1", "candidate"
        )
        self.assertIn(
            member.MEMBER_POLICY_VERSION,
            str(paths["result_dir"]),
        )
        self.assertIn(member.MEMBER_POLICY_VERSION, str(paths["audit"]))
        self.assertIn(member.MEMBER_POLICY_VERSION, str(paths["member_log"]))
        self.assertIn(member.MEMBER_POLICY_VERSION, str(paths["launcher_log"]))
        self.assertNotEqual(
            coarse.command_value(command, "--output-dir"),
            f"runs/{coarse.POLICY_VERSION}/members",
        )

    def test_member_rejects_any_unregistered_dimension(self):
        for bad_alpha in (1.0, 10.0, 20.0, 30.0):
            with self.assertRaisesRegex(ValueError, "Refinement alpha"):
                member.single_run_command(
                    self.root, bad_alpha, 254, "r1", "candidate"
                )
        with self.assertRaisesRegex(ValueError, "budget"):
            member.single_run_command(
                self.root, 5.0, 255, "r1", "candidate"
            )
        with self.assertRaisesRegex(ValueError, "replicate"):
            member.single_run_command(
                self.root, 5.0, 254, "r2", "candidate"
            )

    def test_result_audit_ignores_barrier_array(self):
        size = member.CANONICAL_BUDGET
        common = {
            "active_alpha": np.full(size, 5.0),
            "alpha": np.asarray(5.0),
            "alpha_mode": np.asarray("fixed"),
            "configuration_sha256": np.asarray("a" * 64),
            "i_perp": np.linspace(1.0, 0.01, size),
            "initial_path_sha256": np.asarray("b" * 64),
            "reparam_committed": np.arange(size) % 2 == 0,
            "reparam_mode": np.asarray("state_curve"),
            "segment_length_cv": np.zeros(size),
            "segment_length_ratio": np.ones(size),
            "state_curve_direct_objective_acceptance": np.asarray("per_image"),
            "stop_complete_iperp": np.asarray(-1.0),
            "total_internal_joint_evaluations": np.arange(1, size + 1),
            # Deliberately non-finite: passing proves the audit never indexes it.
            "barrier_kcal": np.full(size, np.nan),
        }
        candidate = {
            **common,
            "discrete_all_returned_states_evaluated": np.asarray(True),
            "discrete_history": np.asarray(7),
            "discrete_implementation_id": np.asarray(
                coarse.EXPECTED_IMPLEMENTATION_ID
            ),
            "discrete_inner_max_steps": np.asarray(10),
            "discrete_residual_ratio": np.asarray(0.8),
            "discrete_stop_mode": np.asarray("r08c10"),
            "frozen_subproblem": np.asarray("discrete_b040"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "candidate.npz"
            np.savez(result, **candidate)
            summary = member._validate_result_iperp_only(
                result, 5.0, "candidate"
            )
        self.assertNotIn("barrier_kcal", summary["result_fields_accessed"])
        self.assertEqual(summary["mesh_eligible_complete_state_count"], 127)

    def test_all_explicit_result_access_sets_exclude_barrier(self):
        for keys in (
            member.COMMON_RESULT_KEYS,
            member.CANDIDATE_RESULT_KEYS,
            member.CONTROL_FORBIDDEN_RESULT_KEYS,
        ):
            self.assertFalse(any("barrier" in key.lower() for key in keys), keys)

    def test_preregistration_hashes_decision_chain_and_run_sources(self):
        planned = self.root / "reports" / (
            f"{grid.PROTOCOL_ID}_N254_r1_unit_test_never_written.json"
        )
        immutable_preregistrations = sorted(
            (self.root / "reports").glob(
                f"{grid.PROTOCOL_ID}_N254_r1_*_preregistered.json"
            )
        )
        if immutable_preregistrations:
            protocol = json.loads(
                immutable_preregistrations[-1].read_text(encoding="utf-8")
            )
        else:
            protocol = grid.build_preregistration(self.root, planned)
        self.assertEqual(protocol["new_alpha_grid"], [5.0, 7.5, 12.5, 15.0])
        self.assertTrue(protocol["all_non_alpha_scientific_flags_frozen"])
        self.assertFalse(protocol["barrier_fields_accessed_by_wrapper_or_launcher"])
        self.assertTrue(protocol["alpha_lock_requires_primary_success"])
        no_dominator = protocol["no_primary_dominator_rule"]
        self.assertFalse(no_dominator["winner_claim_allowed"])
        self.assertFalse(no_dominator["barrier_unembargo_allowed"])
        self.assertEqual(
            no_dominator["eligible_center_alphas"],
            [5.0, 7.5, 10.0, 12.5, 15.0, 20.0],
        )
        self.assertEqual(
            set(protocol["decision_inputs"]),
            {
                "coarse_preregistration",
                "coarse_complete_report",
                "coarse_readonly_iperp_ctp_report",
                "common_initial_path",
                "official_stage4_v4_comparator",
            },
        )
        self.assertEqual(len(protocol["run_source_manifest"]), 8)
        self.assertEqual(
            len(
                protocol["coarse_decision_findings"][
                    "frozen_coarse_source_manifest_verified"
                ]
            ),
            9,
        )
        self.assertEqual(
            protocol["coarse_decision_findings"][
                "coarse_joint_full_curve_dominators"
            ],
            [],
        )
        references = protocol["coarse_decision_findings"][
            "reference_cells_reused_not_rerun"
        ]
        self.assertEqual(set(references), {"alpha10", "alpha20"})
        for pair in references.values():
            self.assertEqual(set(pair), set(member.MEMBER_ROLES))
        if immutable_preregistrations:
            with self.assertRaisesRegex(FileExistsError, "pre-existing"):
                grid.validate_preregistration(protocol, self.root)
        else:
            grid.validate_preregistration(protocol, self.root)

        drifted = copy.deepcopy(protocol)
        drifted["run_source_manifest"]["core_runner"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "Hash drift"):
            grid.validate_preregistration(drifted, self.root)

    def test_preregister_and_launch_cli_are_mutually_exclusive(self):
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
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--root", str(self.root)])
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "--root",
                        str(self.root),
                        "--preregister-only",
                        "--preregistration",
                        "x.json",
                    ]
                )

    def test_no_overwrite_audit_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = member.member_paths(root, 5.0, "candidate")
            audit = paths["audit"]
            self.assertIsInstance(audit, Path)
            audit.parent.mkdir(parents=True)
            audit.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "pre-existing"):
                grid._assert_no_member_outputs(root)


if __name__ == "__main__":
    unittest.main()
