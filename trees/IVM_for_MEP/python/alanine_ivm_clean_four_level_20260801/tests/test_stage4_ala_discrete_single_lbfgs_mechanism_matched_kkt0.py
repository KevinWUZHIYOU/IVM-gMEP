from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import run_ivm_v9_discrete_scalar_lbfgs_shim as scalar_runtime
import run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0 as member
import run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_grid as grid


class MatchedKktZeroMechanismProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def test_three_commands_are_exact_registered_clones(self):
        old = member.canonical_old_current_command(self.root)
        commands = {}
        for arm in member.ARMS:
            command, label = member.single_run_command(self.root, arm)
            member.assert_registered_command(command, old, self.root, arm)
            commands[arm] = command
            self.assertIn(arm, label)
            self.assertIn("a15", label)
            self.assertIn("kkt0", label)
            options = member.command_options(command)
            self.assertEqual(options["--alpha"], "15")
            self.assertEqual(options["--discrete-step-multiplier-eta"], "1.1")
            self.assertEqual(options["--discrete-residual-ratio"], "0.8")
            self.assertEqual(options["--discrete-inner-max-steps"], "10")
            self.assertEqual(options["--discrete-history"], "7")
            self.assertEqual(options["--plbfgs-kkt-atol"], "0")

        current = commands["current_ffrbk_selector"]
        for arm in member.SCALAR_ARMS:
            direct = commands[arm]
            self.assertEqual(
                direct[1],
                str(
                    (
                        self.root
                        / "src/run_ivm_v9_discrete_scalar_lbfgs_shim.py"
                    ).resolve()
                ),
            )
            self.assertEqual(
                member.command_options(direct)[scalar_runtime.MODE_FLAG], arm
            )
            normalized = list(direct[:-2])
            normalized[1] = current[1]
            for flag in member.ROUTING_OPTIONS:
                normalized[normalized.index(flag) + 1] = current[
                    current.index(flag) + 1
                ]
            self.assertEqual(normalized, current)

    def test_current0_parity_contract_keeps_mechanism_ids(self):
        self.assertNotIn("configuration_sha256", member.CURRENT0_PARITY_KEYS)
        self.assertNotIn("plbfgs_kkt_atol", member.CURRENT0_PARITY_KEYS)
        for key in (
            "discrete_implementation_id",
            "discrete_step_multiplier_base_implementation_id",
            "discrete_step_multiplier_implementation_id",
            "discrete_step_multiplier_audit_selection",
            "stored_paths",
            "i_perp",
            "total_internal_joint_evaluations",
        ):
            self.assertIn(key, member.CURRENT0_PARITY_KEYS)

    def test_current0_validator_is_whole_run_parity_and_barrier_blind(self):
        required = (
            member.COMMON_RESULT_KEYS
            | member.DISCRETE_RESULT_KEYS
            | member.ETA_RESULT_KEYS
        )
        with np.load(
            self.root / member.OLD_CURRENT_RESULT, allow_pickle=False
        ) as saved:
            payload = {key: np.array(saved[key], copy=True) for key in required}
        payload["plbfgs_kkt_atol"] = np.asarray(0.0)
        payload["configuration_sha256"] = np.asarray("synthetic-kkt0")
        # Loading this member would fail with allow_pickle=False.  Passing the
        # validator therefore proves it only indexes its explicit allowlist.
        payload["barrier_kcal"] = np.asarray(["embargoed"], dtype=object)
        payload["densified_energy_barrier"] = np.asarray(
            ["embargoed"], dtype=object
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "current0.npz"
            np.savez(result, **payload)
            summary = member.validate_result_iperp_only(
                self.root, result, "current_ffrbk_selector"
            )
        self.assertTrue(summary["old_alpha15_eta1p1_whole_run_safe_parity"])
        self.assertFalse(summary["physical_energy_or_barrier_arrays_accessed"])
        self.assertNotIn("barrier_kcal", summary["result_fields_accessed"])
        self.assertNotIn(
            "densified_energy_barrier", summary["result_fields_accessed"]
        )

    def test_scalar_allowlist_and_runtime_contract_are_explicit(self):
        self.assertEqual(member.CANONICAL_ALPHA, 15.0)
        self.assertEqual(member.CANONICAL_ETA, 1.1)
        self.assertEqual(member.CANONICAL_RATIO, 0.8)
        self.assertEqual(member.CANONICAL_MAX_EVALUATED_STEPS, 10)
        self.assertEqual(member.CANONICAL_HISTORY, 7)
        self.assertEqual(member.CANONICAL_KKT_ATOL, 0.0)
        self.assertEqual(scalar_runtime.FIXED_KKT_ATOL, 0.0)
        self.assertEqual(
            scalar_runtime.VALID_MODES,
            ("scalar_selector", "direct_lbfgs_only"),
        )
        self.assertFalse(
            any(
                "barrier" in key.lower() or "densif" in key.lower()
                for key in member.SCALAR_RESULT_KEYS
            )
        )
        for key in (
            "discrete_scalar_lbfgs_fixed_solver_preconditioner_is_none",
            "discrete_scalar_lbfgs_ffrbk_solve_normal_calls_inside_fixed_solver",
            "discrete_scalar_lbfgs_bank_build_raw_joint_evaluation_delta",
            "discrete_scalar_lbfgs_audit_h0_constructed",
            "discrete_scalar_lbfgs_audit_selector_used",
            "discrete_scalar_lbfgs_audit_direct_fallback",
            "discrete_scalar_lbfgs_audit_gamma_floor_applied",
        ):
            self.assertIn(key, member.SCALAR_RESULT_KEYS)

    def test_preregistration_freezes_neutral_three_cell_decision(self):
        planned = self.root / "reports" / (
            f"{grid.PROTOCOL_ID}_N254_r1_unit_test_never_written.json"
        )
        protocol = grid.build_preregistration(self.root, planned)
        self.assertEqual(protocol["cell_order"], list(member.ARMS))
        self.assertEqual(protocol["fixed_alpha"], 15.0)
        self.assertEqual(protocol["fixed_eta"], 1.1)
        self.assertEqual(protocol["fixed_kkt_atol"], 0.0)
        self.assertTrue(protocol["no_single_arm_predeclared_winner"])
        self.assertEqual(
            protocol["primary_comparators_for_each_performance_candidate"],
            [
                "current_ffrbk_selector",
                "same_alpha_legacy",
                "official_stage4_v4",
            ],
        )
        self.assertFalse(protocol["barrier_unembargo_allowed_by_this_protocol"])
        self.assertFalse(protocol["post_hoc_alpha_eta_or_arm_expansion_allowed"])
        self.assertEqual(
            set(protocol["registered_commands"]), set(member.ARMS)
        )
        self.assertEqual(
            protocol["run_source_manifest"]["scalar_runtime"]["sha256"],
            grid.LOCKED_SOURCE_SHA256["scalar_runtime"],
        )
        grid.validate_preregistration(protocol, self.root)
        drift = copy.deepcopy(protocol)
        drift["fixed_alpha"] = 12.5
        with self.assertRaisesRegex(RuntimeError, "drifted"):
            grid.validate_preregistration(drift, self.root)

    def test_no_overwrite_and_preregister_launch_cli_are_separate(self):
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
