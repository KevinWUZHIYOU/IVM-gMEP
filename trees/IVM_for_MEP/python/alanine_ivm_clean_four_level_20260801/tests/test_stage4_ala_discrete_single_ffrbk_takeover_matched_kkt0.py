from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import run_ivm_v9_discrete_ffrbk_takeover_selector_shim as runtime
import run_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0 as member
import run_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_grid as grid
import run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0 as base
import run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_direct_continuation as flexible
import run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_direct_continuation_grid as prior_grid


class FFrbkTakeoverFormalProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def test_commands_are_exact_current0_clones_except_runtime_mode_and_routing(self):
        current0 = member.current0_command(self.root)
        self.assertEqual(member._file_sha256(self.root / member.CURRENT0_COMMAND),
                         member.CURRENT0_COMMAND_SHA256)
        hashes = set()
        for arm in member.ARMS:
            command, label = member.single_run_command(self.root, arm)
            member.assert_registered_command(command, current0, self.root, arm)
            self.assertEqual(len(command), len(current0) + 2)
            self.assertEqual(command[-2:], [runtime.MODE_FLAG, arm])
            self.assertEqual(
                Path(command[1]).name,
                "run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py",
            )
            differences = [
                index
                for index, (old, new) in enumerate(zip(current0, command[:-2]))
                if old != new
            ]
            self.assertEqual(
                differences,
                sorted(
                    [1]
                    + [
                        current0.index(flag) + 1
                        for flag in sorted(member.ROUTING_OPTIONS)
                    ]
                ),
            )
            options = base.command_options(command)
            self.assertEqual(options["--alpha"], "15")
            self.assertEqual(options["--discrete-step-multiplier-eta"], "1.1")
            self.assertEqual(options["--discrete-residual-ratio"], "0.8")
            self.assertEqual(options["--discrete-inner-max-steps"], "10")
            self.assertEqual(options["--plbfgs-kkt-atol"], "0")
            self.assertEqual(options[runtime.MODE_FLAG], arm)
            self.assertIn(arm, label)
            hashes.add(member._compact_command_sha256(command))
        self.assertEqual(len(hashes), 2)

    def test_flexible_budget_accepts_valid_early_prefix_as_nonconverged(self):
        with np.load(self.root / prior_grid.SCALAR_RESULT, allow_pickle=False) as saved:
            summary = flexible._validate_common_flexible(saved)
        self.assertEqual(summary["trajectory_record_count"], 58)
        self.assertEqual(summary["trajectory_record_count_allowed_range"], [1, 254])
        self.assertTrue(summary["early_stop"])
        self.assertFalse(summary["trajectory_complete_to_budget"])
        self.assertFalse(summary["scientific_run_converged"])
        self.assertTrue(summary["precision_beyond_tightest_reached_unreachable"])
        self.assertEqual(summary["unreachable_threshold_policy"],
                         "candidate_unreachable_is_loss")

    def _synthetic_operator_payload(self) -> dict[str, np.ndarray]:
        result = self.root / grid.PRIOR_ARTIFACTS["matched_current0_result"][0]
        required = (
            base.COMMON_RESULT_KEYS
            | base.DISCRETE_RESULT_KEYS
            | base.ETA_RESULT_KEYS
            | {"stop_reason"}
        )
        with np.load(result, allow_pickle=False) as saved:
            payload = {key: np.array(saved[key], copy=True) for key in required}

        selection = np.asarray(payload["discrete_audit_selection"], dtype="U16")
        step = np.asarray(payload["discrete_audit_step"], dtype=int)
        selection[:] = "LBFGS"
        payload["discrete_audit_selection"] = selection
        payload["discrete_step_multiplier_audit_selection"] = selection.copy()
        payload["discrete_h0_selection_count"] = np.asarray(
            np.count_nonzero(selection == "H0")
        )
        payload["discrete_lbfgs_selection_count"] = np.asarray(
            np.count_nonzero(selection == "LBFGS")
        )
        pre_eta = runtime.selector_implementation_id(
            base.BASE_DISCRETE_IMPLEMENTATION_ID, "forced_operator_lbfgs_selection"
        )
        final = runtime.final_implementation_id(
            base.BASE_DISCRETE_IMPLEMENTATION_ID, "forced_operator_lbfgs_selection"
        )
        payload["discrete_implementation_id"] = np.asarray(final)
        payload["discrete_step_multiplier_base_implementation_id"] = np.asarray(pre_eta)
        payload["discrete_step_multiplier_implementation_id"] = np.asarray(final)

        mode = "forced_operator_lbfgs_selection"
        payload.update(
            {
                "discrete_ffrbk_selector_runtime_policy_id": np.asarray(
                    runtime.RUNTIME_POLICY_ID
                ),
                "discrete_ffrbk_selector_mode_policy_id": np.asarray(
                    runtime.MODE_POLICY_IDS[mode]
                ),
                "discrete_ffrbk_selector_mode": np.asarray(mode),
                "discrete_ffrbk_selector_operator_policy_id": np.asarray(
                    runtime.OPERATOR_POLICY_ID
                ),
                "discrete_ffrbk_selector_base_implementation_id": np.asarray(
                    base.BASE_DISCRETE_IMPLEMENTATION_ID
                ),
                "discrete_ffrbk_selector_pre_eta_implementation_id": np.asarray(pre_eta),
                "discrete_ffrbk_selector_final_implementation_id": np.asarray(final),
                "discrete_ffrbk_selector_alpha": np.asarray(base.CANONICAL_ALPHA),
                "discrete_ffrbk_selector_eta": np.asarray(base.CANONICAL_ETA),
                "discrete_ffrbk_selector_stop_mode": np.asarray("r08c10"),
                "discrete_ffrbk_selector_residual_ratio": np.asarray(
                    base.CANONICAL_RATIO
                ),
                "discrete_ffrbk_selector_inner_max_steps": np.asarray(
                    base.CANONICAL_MAX_EVALUATED_STEPS
                ),
                "discrete_ffrbk_selector_history": np.asarray(
                    base.CANONICAL_HISTORY
                ),
                "discrete_ffrbk_selector_kkt_atol": np.asarray(
                    base.CANONICAL_KKT_ATOL
                ),
                "discrete_ffrbk_selector_fixed_solver_calls": np.asarray(
                    np.unique(payload["discrete_audit_outer"]).size
                ),
                "discrete_ffrbk_selector_latch_reset_count": np.asarray(
                    np.unique(payload["discrete_audit_outer"]).size
                ),
                "discrete_ffrbk_selector_aborted_solver_calls": np.asarray(0),
                "discrete_ffrbk_selector_apply_h0_inverse_calls": np.asarray(
                    selection.size
                ),
                "discrete_ffrbk_selector_apply_h0_inverse_preconditioner_none_calls": (
                    np.asarray(0)
                ),
                "discrete_ffrbk_selector_joint_lbfgs_direction_calls": np.asarray(
                    selection.size
                ),
                "discrete_ffrbk_selector_committed_selection_count": np.asarray(
                    selection.size
                ),
                "discrete_ffrbk_selector_uncommitted_selection_count": np.asarray(0),
                "discrete_ffrbk_selector_aborted_selection_count": np.asarray(0),
                "discrete_ffrbk_selector_energy_backtracking": np.asarray(False),
                "discrete_ffrbk_selector_extra_physical_evaluations": np.asarray(0),
                "discrete_ffrbk_selector_operator_preserved": np.asarray(True),
                "discrete_ffrbk_selector_post_select_eta_composed": np.asarray(True),
                "discrete_ffrbk_selector_reparameterization_inside_subproblem": (
                    np.asarray(False)
                ),
                "discrete_ffrbk_selector_selector_only_change": np.asarray(True),
                "discrete_ffrbk_selector_h0_shadow_constructed": np.asarray(True),
                "discrete_ffrbk_selector_true_lbfgs_only_construction": np.asarray(
                    False
                ),
                "discrete_ffrbk_selector_invalid_lbfgs_h0_fallback_enabled": (
                    np.asarray(True)
                ),
                "discrete_ffrbk_selector_one_way_latch_is_preference": np.asarray(
                    False
                ),
                "discrete_ffrbk_selector_audit_selection": selection.copy(),
                "discrete_ffrbk_selector_audit_solver_call": np.asarray(
                    payload["discrete_audit_outer"], dtype=int
                ),
                "discrete_ffrbk_selector_audit_selector_call": np.asarray(
                    payload["discrete_audit_step"], dtype=int
                ),
                "discrete_ffrbk_selector_audit_latched_before": np.zeros(
                    selection.shape, dtype=bool
                ),
                "discrete_ffrbk_selector_audit_latched_after": np.zeros(
                    selection.shape, dtype=bool
                ),
                "discrete_ffrbk_selector_audit_strict_win_trigger": np.zeros(
                    selection.shape, dtype=bool
                ),
                "discrete_ffrbk_selector_audit_forced_lbfgs": selection == "LBFGS",
                "discrete_ffrbk_selector_audit_empty_history_tie": step == 0,
                "discrete_ffrbk_selector_audit_h0_failover": np.zeros(
                    selection.shape, dtype=bool
                ),
                "discrete_ffrbk_selector_audit_h0_valid": np.ones(
                    selection.shape, dtype=bool
                ),
                "discrete_ffrbk_selector_audit_lbfgs_valid": np.ones(
                    selection.shape, dtype=bool
                ),
                "discrete_ffrbk_selector_audit_h0_capped_decrease": np.ones(
                    selection.shape, dtype=float
                ),
                "discrete_ffrbk_selector_audit_lbfgs_capped_decrease": np.ones(
                    selection.shape, dtype=float
                ),
            }
        )
        return payload

    def test_result_validator_is_iperp_only_and_checks_operator_mode(self):
        payload = self._synthetic_operator_payload()
        payload["densified_energy_barrier"] = np.asarray(["embargoed"], dtype=object)
        payload["barrier_kcal"] = np.asarray(["embargoed"], dtype=object)
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "operator.npz"
            np.savez(result, **payload)
            summary = member.validate_result_iperp_only(
                self.root, result, "forced_operator_lbfgs_selection"
            )
            payload["discrete_ffrbk_selector_operator_preserved"] = np.asarray(False)
            invalid = Path(temporary) / "invalid.npz"
            np.savez(invalid, **payload)
            with self.assertRaisesRegex(RuntimeError, "boolean drifted"):
                member.validate_result_iperp_only(
                    self.root, invalid, "forced_operator_lbfgs_selection"
                )
        self.assertFalse(summary["physical_energy_or_barrier_arrays_accessed"])
        self.assertNotIn("densified_energy_barrier", summary["result_fields_accessed"])
        self.assertNotIn("barrier_kcal", summary["result_fields_accessed"])
        self.assertEqual(summary["trajectory_record_count"], 254)

    def test_one_empty_history_selection_allows_zero_joint_lbfgs_calls(self):
        payload = self._synthetic_operator_payload()
        original_steps = np.asarray(payload["discrete_audit_selection"]).size
        for key, value in list(payload.items()):
            array = np.asarray(value)
            if (
                array.ndim > 0
                and array.shape[0] == original_steps
                and (
                    key.startswith("discrete_audit_")
                    or key.startswith("discrete_step_multiplier_audit_")
                    or key.startswith("discrete_ffrbk_selector_audit_")
                )
            ):
                payload[key] = array[:1].copy()
        payload["discrete_audit_stop_reason"] = np.asarray(
            payload["discrete_audit_stop_reason"]
        )[:1].copy()
        payload["discrete_audit_global_relative_residual"] = np.asarray(
            payload["discrete_audit_global_relative_residual"]
        )[:1].copy()
        payload["discrete_h0_selection_count"] = np.asarray(0)
        payload["discrete_lbfgs_selection_count"] = np.asarray(1)
        payload["discrete_ffrbk_selector_fixed_solver_calls"] = np.asarray(1)
        payload["discrete_ffrbk_selector_latch_reset_count"] = np.asarray(1)
        payload["discrete_ffrbk_selector_apply_h0_inverse_calls"] = np.asarray(1)
        payload["discrete_ffrbk_selector_joint_lbfgs_direction_calls"] = np.asarray(0)
        payload["discrete_ffrbk_selector_committed_selection_count"] = np.asarray(1)
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "empty-history-only.npz"
            np.savez(result, **payload)
            summary = member.validate_result_iperp_only(
                self.root, result, "forced_operator_lbfgs_selection"
            )
        self.assertEqual(summary["apply_h0_inverse_call_count"], 1)
        self.assertEqual(summary["joint_lbfgs_direction_call_count"], 0)

    def test_preregistration_pins_controls_ctp_sources_and_jobs1(self):
        planned = self.root / "reports" / (
            f"{grid.PROTOCOL_ID}_N254_r1_unit_test_never_written.json"
        )
        protocol = grid.build_preregistration(self.root, planned)
        self.assertEqual(protocol["cell_order"], list(member.ARMS))
        self.assertEqual(protocol["trajectory_record_count_allowed_range"], [1, 254])
        self.assertFalse(protocol["early_stop_counted_as_converged"])
        self.assertTrue(protocol["identical_candidate_trajectories_allowed"])
        self.assertTrue(
            protocol["forced_operator_lbfgs_is_selection_not_computation_ablation"]
        )
        self.assertIn("shadow", protocol["forced_operator_lbfgs_h0_shadow_policy"])
        self.assertIn("without clearing", protocol["one_way_invalid_lbfgs_safety_fallback"])
        self.assertEqual(protocol["execution_jobs_fixed"], 1)
        evidence = protocol["mechanism_ctp_evidence"]
        self.assertEqual(evidence["performance_metric"], "I_perp_only")
        self.assertFalse(evidence["physical_objective_arrays_accessed"])
        self.assertFalse(evidence["densification_performed"])
        for key in (
            "mechanism_ctp_readonly",
            "matched_current0_result",
            "old_alpha15_eta1p1_result",
            "same_alpha_legacy_result",
            "official_stage4_v4",
        ):
            self.assertIn(key, protocol["prior_artifact_manifest"])
        self.assertEqual(
            protocol["prior_artifact_manifest"]["mechanism_ctp_readonly"]["sha256"],
            grid.MECHANISM_CTP_REPORT_SHA256,
        )
        for key in (
            "member",
            "grid",
            "campaign_test",
            "takeover_runtime",
            "takeover_runtime_test",
            "frozen_discrete",
            "production_runner",
        ):
            self.assertIn(key, protocol["run_source_manifest"])
        grid.validate_preregistration(protocol, self.root)
        drift = copy.deepcopy(protocol)
        drift["trajectory_record_count_allowed_range"] = [254, 254]
        with self.assertRaisesRegex(RuntimeError, "drifted"):
            grid.validate_preregistration(drift, self.root)

    def test_no_overwrite_exclusive_write_and_split_actions(self):
        grid._assert_no_outputs(self.root)
        with tempfile.TemporaryDirectory() as temporary:
            fake_root = Path(temporary)
            paths = member.member_paths(fake_root, member.ARMS[0])
            audit = paths["audit"]
            assert isinstance(audit, Path)
            audit.parent.mkdir(parents=True)
            audit.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "pre-existing"):
                grid._assert_no_outputs(fake_root)
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
                str(self.root / "reports/registered.json"),
            ]
        )
        self.assertTrue(prereg.preregister_only)
        self.assertIsNone(prereg.preregistration)
        self.assertFalse(launch.preregister_only)
        self.assertIsNotNone(launch.preregistration)


if __name__ == "__main__":
    unittest.main()
