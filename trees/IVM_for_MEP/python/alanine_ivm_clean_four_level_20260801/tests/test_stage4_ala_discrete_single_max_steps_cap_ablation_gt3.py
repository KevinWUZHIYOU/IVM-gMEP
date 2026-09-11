from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import run_stage4_ala_discrete_single_max_steps_cap_ablation_gt3 as member
import run_stage4_ala_discrete_single_max_steps_cap_ablation_gt3_grid as grid


class MaxEvaluatedStepsCapAblationGT3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def test_caps_tags_and_primary_set_obey_updated_hard_constraint(self):
        self.assertEqual(member.NEW_CAPS, (4, 5, 6))
        self.assertTrue(all(cap > 3 for cap in member.NEW_CAPS))
        self.assertEqual(
            [member.cap_tag(cap) for cap in member.NEW_CAPS],
            ["c04", "c05", "c06"],
        )
        self.assertEqual(member.REFERENCE_CAP, 10)
        self.assertEqual(member.PRIMARY_LOCK_ELIGIBLE_CAPS, (4, 5, 6, 10))
        self.assertEqual(member.CANONICAL_ALPHA, 12.5)
        self.assertEqual(member.CANONICAL_RATIO, 0.8)
        self.assertEqual(member.CANONICAL_BUDGET, 254)
        self.assertEqual(member.LBFGS_HISTORY, 7)
        with self.assertRaisesRegex(ValueError, "one of 4,5,6"):
            member.single_run_command(self.root, 3)

    def test_commands_differ_from_cap10_only_by_routing_and_cap(self):
        canonical = member.canonical_reference_command(self.root)
        canonical_options = member.superseded_v1.coarse_single.command_options(canonical)
        for cap in member.NEW_CAPS:
            command, label = member.single_run_command(self.root, cap)
            member.assert_cap_only_scientific_change(command, canonical, cap)
            options = member.superseded_v1.coarse_single.command_options(command)
            changed_indices = {
                index
                for index, (observed, expected) in enumerate(zip(command, canonical))
                if observed != expected
            }
            expected_changed_indices = {
                canonical.index(flag) + 1
                for flag in member.ROUTING_OPTIONS
                | {member.SCIENTIFIC_ABLATION_OPTION}
            }
            self.assertEqual(changed_indices, expected_changed_indices)
            self.assertEqual(options["--discrete-inner-max-steps"], str(cap))
            self.assertEqual(options["--alpha"], "12.5")
            self.assertEqual(options["--alpha-mode"], "fixed")
            self.assertEqual(options["--discrete-residual-ratio"], "0.8")
            self.assertEqual(options["--discrete-stop-mode"], "r08c10")
            self.assertEqual(options["--discrete-history"], "7")
            self.assertEqual(options["--history"], "7")
            self.assertEqual(options["--budget"], "254")
            self.assertEqual(options["--reparam-mode"], "state_curve")
            self.assertEqual(
                options["--state-curve-direct-objective-acceptance"],
                "per_image",
            )
            self.assertIn(member.cap_tag(cap), label)
            for flag, expected in canonical_options.items():
                if flag in member.ROUTING_OPTIONS | {
                    member.SCIENTIFIC_ABLATION_OPTION
                }:
                    continue
                self.assertEqual(options[flag], expected, flag)

    def _synthetic_result(self, cap: int) -> dict[str, np.ndarray]:
        size = member.CANONICAL_BUDGET
        steps = np.arange(cap, dtype=int)
        selections = np.asarray(["H0"] + ["LBFGS"] * (cap - 1))
        h0 = int(np.count_nonzero(selections == "H0"))
        lbfgs = int(np.count_nonzero(selections == "LBFGS"))
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
            "discrete_audit_history_count": np.arange(cap),
            "discrete_audit_outer": np.zeros(cap, dtype=int),
            "discrete_audit_selection": selections,
            "discrete_audit_step": steps,
            "discrete_audit_stop_reason": np.asarray(
                ["evaluated-inner-max-steps"]
            ),
            "discrete_history": np.asarray(7),
            "discrete_h0_selection_count": np.asarray(h0),
            "discrete_implementation_id": np.asarray(
                member.superseded_v1.coarse_single.EXPECTED_IMPLEMENTATION_ID
            ),
            "discrete_inner_max_steps": np.asarray(cap),
            "discrete_lbfgs_selection_count": np.asarray(lbfgs),
            "discrete_residual_ratio": np.asarray(0.8),
            "discrete_stop_mode": np.asarray("r08c10"),
            "frozen_subproblem": np.asarray("discrete_b040"),
            # Object dtype would fail with allow_pickle=False if indexed.  Its
            # presence therefore audits that the embargoed field stays unread.
            "barrier_kcal": np.asarray(["embargoed"], dtype=object),
        }

    def test_each_new_cap_validator_observes_lbfgs_without_reading_barrier(self):
        with tempfile.TemporaryDirectory() as temporary:
            for cap in member.NEW_CAPS:
                result = Path(temporary) / f"cap{cap}.npz"
                np.savez(result, **self._synthetic_result(cap))
                summary = member._validate_result_iperp_only(result, cap)
                self.assertGreater(summary["lbfgs_selection_count"], 0)
                self.assertTrue(summary["lbfgs_takeover_observed"])
                self.assertNotIn("barrier_kcal", summary["result_fields_accessed"])

    def test_result_validator_rejects_negative_residual_audit(self):
        payload = self._synthetic_result(4)
        payload["discrete_audit_global_relative_residual"] = np.asarray([-0.1])
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "negative_residual.npz"
            np.savez(result, **payload)
            with self.assertRaisesRegex(RuntimeError, "History/residual"):
                member._validate_result_iperp_only(result, 4)

    def test_result_access_allowlists_exclude_every_barrier_field(self):
        for keys in (member.COMMON_RESULT_KEYS, member.DISCRETE_RESULT_KEYS):
            self.assertFalse(any("barrier" in key.lower() for key in keys), keys)

    def test_superseded_v1_files_are_byte_identical_and_have_zero_outputs(self):
        expected = {
            **grid.SUPERSEDED_SOURCE_HASHES,
            "preregistration": grid.SUPERSEDED_PREREGISTRATION_SHA256,
        }
        paths = {
            **{
                name: self.root / path
                for name, path in grid.SUPERSEDED_SOURCE_PATHS.items()
            },
            "preregistration": self.root / grid.SUPERSEDED_PREREGISTRATION,
        }
        self.assertEqual(
            {name: grid.file_sha256(path) for name, path in paths.items()},
            expected,
        )
        evidence = grid._superseded_output_evidence(self.root)
        self.assertEqual(evidence["run_file_count"], 0)
        self.assertEqual(evidence["log_file_count"], 0)
        self.assertFalse(evidence["planned_report_exists"])
        grid._assert_no_outputs(self.root)

    def test_supersession_sidecar_locks_reason_hashes_and_zero_output_proof(self):
        sidecar = grid.build_supersession_payload(self.root)
        self.assertEqual(sidecar["status"], "superseded_unlaunched")
        self.assertEqual(sidecar["new_hard_constraint"], "max_evaluated_steps > 3")
        self.assertEqual(sidecar["reason"], grid.SUPERSESSION_REASON)
        self.assertEqual(
            sidecar["reason"],
            "user objective updated maxstep>3 before any launch",
        )
        self.assertEqual(
            sidecar["superseded_preregistration"]["sha256"],
            grid.SUPERSEDED_PREREGISTRATION_SHA256,
        )
        self.assertEqual(sidecar["zero_output_evidence"]["run_file_count"], 0)
        self.assertEqual(sidecar["zero_output_evidence"]["log_file_count"], 0)
        self.assertFalse(
            sidecar["zero_output_evidence"]["planned_report_exists"]
        )
        self.assertFalse(sidecar["barrier_data_accessed"])

    def test_supersession_sidecar_verifier_fails_closed_on_semantic_drift(self):
        for key, invalid, message in (
            ("reason", "different reason", "reason drifted"),
            ("superseded_protocol_id", "different", "Invalid supersession"),
            (
                "superseded_preregistration_remains_immutable",
                False,
                "no longer preserves",
            ),
            ("barrier_data_accessed", True, "barrier embargo"),
        ):
            payload = grid.build_supersession_payload(self.root)
            payload[key] = invalid
            with tempfile.TemporaryDirectory() as temporary:
                sidecar = Path(temporary) / "mutated.json"
                grid._write_json_exclusive(sidecar, payload)
                with self.assertRaisesRegex(RuntimeError, message):
                    grid._verify_supersession_sidecar(
                        grid._hashed(sidecar), self.root
                    )

    def _build_preregistration_with_temporary_sidecar(
        self,
    ) -> tuple[dict[str, object], Path, tempfile.TemporaryDirectory]:
        temporary = tempfile.TemporaryDirectory()
        temporary_path = Path(temporary.name)
        sidecar_path = temporary_path / "supersession.json"
        grid._write_json_exclusive(
            sidecar_path, grid.build_supersession_payload(self.root)
        )
        sidecar_record = grid._hashed(sidecar_path)
        planned = self.root / "reports" / (
            f"{grid.PROTOCOL_ID}_N254_r1_unit_test_never_written.json"
        )
        protocol = grid.build_preregistration(
            self.root, planned, sidecar_record
        )
        return protocol, sidecar_path, temporary

    def test_preregistration_hashes_full_chain_and_freezes_primary_rule(self):
        protocol, sidecar_path, temporary = (
            self._build_preregistration_with_temporary_sidecar()
        )
        try:
            self.assertEqual(protocol["new_max_evaluated_steps_caps"], [4, 5, 6])
            self.assertEqual(protocol["display_caps"], [4, 5, 6, 10])
            self.assertEqual(protocol["new_member_tags"], ["c04", "c05", "c06"])
            self.assertEqual(protocol["primary_lock_eligible_caps"], [4, 5, 6, 10])
            self.assertIn(
                "at least one observed LBFGS selection",
                protocol["primary_success_criterion"],
            )
            self.assertIn("both comparators", protocol["primary_success_criterion"])
            self.assertEqual(
                protocol["supersession_sidecar"]["sha256"],
                grid.file_sha256(sidecar_path),
            )
            self.assertEqual(
                protocol["superseded_v1_preregistration"]["sha256"],
                grid.SUPERSEDED_PREREGISTRATION_SHA256,
            )
            self.assertEqual(
                protocol["formal_ratio_chain"]["formal_ratio_analyzer_output"][
                    "sha256"
                ],
                grid.superseded_grid.FORMAL_RATIO_ANALYSIS_SHA256,
            )
            self.assertEqual(
                protocol["canonical_common_initial_path"]["sha256"],
                grid.superseded_grid.COMMON_INITIAL_FILE_SHA256,
            )
            self.assertEqual(
                protocol["canonical_common_initial_path"][
                    "initial_path_array_sha256"
                ],
                grid.superseded_grid.COMMON_INITIAL_ARRAY_SHA256,
            )
            self.assertEqual(
                protocol["reference_mechanism_evidence"][
                    "per_solve_evaluated_step_count_distribution"
                ],
                {"1": 4, "3": 12, "4": 3, "6": 1},
            )
            self.assertEqual(
                set(
                    protocol["reference_mechanism_evidence"][
                        "gt3_ablation_rationale"
                    ]
                ),
                {"cap4", "cap5", "cap6"},
            )
            self.assertNotIn(
                "ablation_rationale",
                protocol["reference_mechanism_evidence"],
            )
            self.assertFalse(
                protocol["no_primary_dominator_rule"]["winner_claim_allowed"]
            )
            self.assertFalse(
                protocol["no_primary_dominator_rule"]["cap_lock_allowed"]
            )
            self.assertFalse(
                protocol["no_primary_dominator_rule"][
                    "post_hoc_grid_expansion_allowed"
                ]
            )
            self.assertFalse(
                protocol["no_primary_dominator_rule"]["barrier_unembargo_allowed"]
            )
            self.assertFalse(protocol["barrier_fields_accessed_by_wrapper_or_launcher"])
            self.assertIn("gt3_launcher", protocol["run_source_manifest"])
            self.assertIn("gt3_member_wrapper", protocol["run_source_manifest"])
            self.assertIn(
                "inherited_core_runner", protocol["run_source_manifest"]
            )
            grid.validate_preregistration(protocol, self.root)

            drifted = copy.deepcopy(protocol)
            drifted["formal_ratio_chain"]["formal_ratio_analyzer_output"][
                "sha256"
            ] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "Formal ratio chain drifted"):
                grid.validate_preregistration(drifted, self.root)
        finally:
            temporary.cleanup()

    def test_exclusive_writes_paths_and_preregister_launch_cli_are_separate(self):
        paths = [member.member_paths(self.root, cap) for cap in member.NEW_CAPS]
        self.assertEqual(len({str(path["label"]) for path in paths}), 3)
        for path in paths:
            self.assertIn(member.MEMBER_POLICY_VERSION, str(path["result_dir"]))
            self.assertNotIn(grid.SUPERSEDED_POLICY_VERSION, str(path["result_dir"]))

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "exclusive.json"
            grid._write_json_exclusive(target, {"first": True})
            with self.assertRaises(FileExistsError):
                grid._write_json_exclusive(target, {"second": True})
            self.assertEqual(json.loads(target.read_text()), {"first": True})

        parser = grid.argument_parser()
        prereg = parser.parse_args(
            ["--root", str(self.root), "--preregister-only"]
        )
        launch = parser.parse_args(
            [
                "--root",
                str(self.root),
                "--preregistration",
                str(self.root / "reports/x.json"),
            ]
        )
        self.assertTrue(prereg.preregister_only)
        self.assertIsNone(prereg.preregistration)
        self.assertFalse(launch.preregister_only)
        self.assertIsNotNone(launch.preregistration)


if __name__ == "__main__":
    unittest.main()
