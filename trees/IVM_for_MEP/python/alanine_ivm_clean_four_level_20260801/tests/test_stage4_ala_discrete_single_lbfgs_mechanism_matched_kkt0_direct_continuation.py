from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import run_ivm_v9_discrete_scalar_lbfgs_shim as scalar_runtime
import run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0 as base
import run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_direct_continuation as member
import run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_direct_continuation_grid as grid


class DirectContinuationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def test_command_is_original_registered_direct_with_routing_only(self):
        original = member._original_direct_command(self.root)
        self.assertEqual(
            member._compact_command_sha256(original),
            member.ORIGINAL_DIRECT_COMMAND_SHA256,
        )
        command, label = member.single_run_command(self.root)
        member.assert_registered_command(command, original, self.root)
        differences = {
            original[index - 1]
            for index, (old, new) in enumerate(
                zip(original, command)
            )
            if old != new
        }
        self.assertEqual(differences, set(member.ROUTING_OPTIONS))
        self.assertIn("direct_continuation_v1", label)
        self.assertEqual(
            base.command_options(command)["--discrete-scalar-lbfgs-mode"],
            "direct_lbfgs_only",
        )

    def test_flexible_validator_accepts_pinned_scalar_early_trajectory(self):
        with np.load(self.root / grid.SCALAR_RESULT, allow_pickle=False) as saved:
            summary = member._validate_common_flexible(saved)
        self.assertEqual(summary["trajectory_record_count"], 58)
        self.assertEqual(summary["final_optimization_NF"], 1160)
        self.assertEqual(summary["physical_stop_reason"], "cached_coordinate_stall")
        self.assertTrue(summary["early_stop"])
        self.assertFalse(summary["trajectory_complete_to_budget"])
        self.assertFalse(summary["scientific_run_converged"])
        self.assertTrue(summary["precision_beyond_tightest_reached_unreachable"])
        self.assertEqual(
            summary["unreachable_threshold_policy"],
            "candidate_unreachable_is_loss",
        )

    def _synthetic_direct_payload(self) -> dict[str, np.ndarray]:
        required = (
            base.COMMON_RESULT_KEYS
            | base.DISCRETE_RESULT_KEYS
            | base.SCALAR_RESULT_KEYS
            | {"stop_reason"}
        )
        with np.load(self.root / grid.SCALAR_RESULT, allow_pickle=False) as saved:
            payload = {key: np.array(saved[key], copy=True) for key in required}
        expected_id = scalar_runtime.shim_implementation_id(
            base.BASE_DISCRETE_IMPLEMENTATION_ID, member.ARM
        )
        payload["discrete_implementation_id"] = np.asarray(expected_id)
        payload["discrete_scalar_lbfgs_mode"] = np.asarray(member.ARM)
        payload["discrete_scalar_lbfgs_mode_policy_id"] = np.asarray(
            scalar_runtime.MODE_POLICY_IDS[member.ARM]
        )
        payload["discrete_scalar_lbfgs_implementation_id"] = np.asarray(
            expected_id
        )
        payload["discrete_scalar_lbfgs_selector_competition"] = np.asarray(False)
        payload["discrete_scalar_lbfgs_direct_only"] = np.asarray(True)
        selection = np.full(
            np.asarray(payload["discrete_audit_selection"]).shape,
            "LBFGS",
            dtype="U8",
        )
        payload["discrete_audit_selection"] = selection
        payload["discrete_scalar_lbfgs_audit_selection"] = selection.copy()
        payload["discrete_scalar_lbfgs_audit_h0_constructed"] = np.zeros(
            selection.shape, dtype=bool
        )
        payload["discrete_scalar_lbfgs_audit_selector_used"] = np.zeros(
            selection.shape, dtype=bool
        )
        payload["discrete_h0_selection_count"] = np.asarray(0)
        payload["discrete_lbfgs_selection_count"] = np.asarray(selection.size)
        return payload

    def test_full_direct_validator_keeps_mechanism_gates_and_barrier_embargo(self):
        payload = self._synthetic_direct_payload()
        payload["barrier_kcal"] = np.asarray(["embargoed"], dtype=object)
        payload["densified_energy_barrier"] = np.asarray(
            ["embargoed"], dtype=object
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "direct.npz"
            np.savez(result, **payload)
            summary = member.validate_result_iperp_only(self.root, result)
            payload["discrete_scalar_lbfgs_aborted_solver_call_count"] = np.asarray(1)
            invalid = Path(temporary) / "aborted.npz"
            np.savez(invalid, **payload)
            with self.assertRaisesRegex(RuntimeError, "forbidden counter"):
                member.validate_result_iperp_only(self.root, invalid)
        self.assertEqual(summary["trajectory_record_count"], 58)
        self.assertEqual(summary["h0_selection_count"], 0)
        self.assertEqual(summary["lbfgs_selection_count"], 80)
        self.assertFalse(summary["physical_energy_or_barrier_arrays_accessed"])
        self.assertNotIn("barrier_kcal", summary["result_fields_accessed"])
        self.assertNotIn(
            "densified_energy_barrier", summary["result_fields_accessed"]
        )

    def test_preregistration_pins_original_failure_evidence_and_sources(self):
        planned = self.root / "reports" / (
            f"{grid.PROTOCOL_ID}_N254_r1_unit_test_never_written.json"
        )
        protocol = grid.build_preregistration(self.root, planned)
        self.assertEqual(protocol["cell_order"], ["direct_lbfgs_only"])
        self.assertEqual(
            protocol["registered_command"]["original_v1_compact_sha256"],
            member.ORIGINAL_DIRECT_COMMAND_SHA256,
        )
        self.assertEqual(
            protocol["original_v1_outcome"][
                "scalar_selector_trajectory_record_count"
            ],
            58,
        )
        self.assertTrue(
            protocol["original_v1_outcome"]["direct_lbfgs_only_was_not_launched"]
        )
        for key in (
            "original_v1_preregistration",
            "original_v1_report",
            "current0_result",
            "current0_member_audit",
            "current0_companion_audit",
            "scalar_selector_result",
            "scalar_selector_companion_audit",
            "scalar_launcher_stderr_log",
        ):
            self.assertIn(key, protocol["prior_artifact_manifest"])
        for key in (
            "continuation_member",
            "continuation_grid",
            "continuation_test",
            "original_v1_scalar_runtime",
            "original_v1_production_runner",
            "original_v1_ala_core",
        ):
            self.assertIn(key, protocol["run_source_manifest"])
        grid.validate_preregistration(protocol, self.root)
        drift = copy.deepcopy(protocol)
        drift["trajectory_record_count_allowed_range"] = [254, 254]
        with self.assertRaisesRegex(RuntimeError, "drifted"):
            grid.validate_preregistration(drift, self.root)

    def test_no_overwrite_and_preregister_launch_are_separate(self):
        grid._assert_no_outputs(self.root)
        with tempfile.TemporaryDirectory() as temporary:
            fake_root = Path(temporary)
            paths = member.member_paths(fake_root)
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
