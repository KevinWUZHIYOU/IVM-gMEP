from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import run_ivm_v9_discrete_selected_eta_shim as eta_shim
import run_stage4_ala_discrete_single_eta1p1_alpha20_followup as member
import run_stage4_ala_discrete_single_eta1p1_alpha20_followup_grid as grid


class EtaOnePointOneAlpha20FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def test_fixed_scientific_configuration_and_exact_clone(self):
        self.assertEqual(member.ETA, 1.1)
        self.assertEqual(member.CANONICAL_ALPHA, 20.0)
        self.assertEqual(member.CANONICAL_RATIO, 0.8)
        self.assertEqual(member.CANONICAL_MAX_EVALUATED_STEPS, 10)
        self.assertEqual(member.CANONICAL_HISTORY, 7)
        self.assertEqual(member.CANONICAL_BUDGET, 254)
        canonical = member.canonical_reference_command(self.root)
        fixedalpha_v2, _ = member.coarse_single.single_run_command(
            self.root, 20.0, 254, "r1", "candidate"
        )
        self.assertEqual(canonical, fixedalpha_v2)
        command, label = member.single_run_command(self.root)
        member.assert_eta1p1_only_scientific_change(command, canonical, self.root)
        self.assertIn("a20", label)
        self.assertNotIn("a10", label)
        self.assertEqual(command[-2:], [eta_shim.ETA_FLAG, "1.1"])
        observed = list(command[:-2])
        observed[1] = canonical[1]
        for flag in member.ROUTING_OPTIONS:
            observed[observed.index(flag) + 1] = canonical[
                canonical.index(flag) + 1
            ]
        self.assertEqual(observed, canonical)

    def _synthetic_result(self) -> dict[str, np.ndarray]:
        required = (
            member.COMMON_RESULT_KEYS
            | member.DISCRETE_RESULT_KEYS
        )
        with np.load(self.root / grid.ETA_ONE_RESULT, allow_pickle=False) as saved:
            payload = {key: np.array(saved[key], copy=True) for key in required}
        implementation_id = eta_shim.shim_implementation_id(
            member.coarse_single.EXPECTED_IMPLEMENTATION_ID, member.ETA
        )
        payload["discrete_implementation_id"] = np.asarray(implementation_id)
        selection = np.asarray(payload["discrete_audit_selection"], dtype="U8")
        size = selection.size
        base = np.full(size, 0.1)
        dynamic = np.full(size, 1.0)
        requested_max = member.ETA * base
        payload.update(
            {
                "discrete_step_multiplier_policy_id": np.asarray(
                    eta_shim.SHIM_POLICY_ID
                ),
                "discrete_step_multiplier_base_implementation_id": np.asarray(
                    member.coarse_single.EXPECTED_IMPLEMENTATION_ID
                ),
                "discrete_step_multiplier_implementation_id": np.asarray(
                    implementation_id
                ),
                "discrete_step_multiplier_eta": np.asarray(member.ETA),
                "discrete_step_multiplier_application_stage": np.asarray(
                    eta_shim.APPLICATION_STAGE
                ),
                "discrete_step_multiplier_selector_isolated": np.asarray(True),
                "discrete_step_multiplier_dynamic_cap_reapplied": np.asarray(True),
                "discrete_step_multiplier_energy_backtracking": np.asarray(False),
                "discrete_step_multiplier_extra_physical_evaluations": np.asarray(0),
                "discrete_step_multiplier_audit_selection": selection.copy(),
                "discrete_step_multiplier_audit_requested": np.full(
                    size, member.ETA
                ),
                "discrete_step_multiplier_audit_applied": np.full(
                    size, member.ETA
                ),
                "discrete_step_multiplier_audit_cap_hit": np.zeros(
                    size, dtype=bool
                ),
                "discrete_step_multiplier_audit_dynamic_max_step": dynamic,
                "discrete_step_multiplier_audit_base_max_atomic_displacement": base,
                "discrete_step_multiplier_audit_requested_max_atomic_displacement": (
                    requested_max
                ),
                "discrete_step_multiplier_audit_applied_max_atomic_displacement": (
                    requested_max.copy()
                ),
                "discrete_step_multiplier_uncommitted_selection_count": np.asarray(0),
                "discrete_step_multiplier_aborted_selection_count": np.asarray(0),
                # Loading this object array would fail with allow_pickle=False.
                "barrier_kcal": np.asarray(["embargoed"], dtype=object),
            }
        )
        return payload

    def test_validator_accepts_eta1p1_and_never_indexes_barrier(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "synthetic.npz"
            np.savez(result, **self._synthetic_result())
            summary = member._validate_result_iperp_only(result)
        self.assertEqual(summary["trajectory_record_count"], 254)
        self.assertEqual(summary["eta_applied_multiplier_min"], 1.1)
        self.assertEqual(summary["eta_applied_multiplier_max"], 1.1)
        self.assertNotIn("barrier_kcal", summary["result_fields_accessed"])

    def test_validator_rejects_energy_backtracking(self):
        payload = self._synthetic_result()
        payload["discrete_step_multiplier_energy_backtracking"] = np.asarray(True)
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "invalid.npz"
            np.savez(result, **payload)
            with self.assertRaisesRegex(RuntimeError, "forbidden energy backtracking"):
                member._validate_result_iperp_only(result)

    def test_result_allowlists_exclude_every_barrier_field(self):
        for keys in (
            member.COMMON_RESULT_KEYS,
            member.DISCRETE_RESULT_KEYS,
            member.ETA_RESULT_KEYS,
        ):
            self.assertFalse(any("barrier" in key.lower() for key in keys))

    def test_preregistration_pins_controls_snapshot_sources_and_command(self):
        planned = self.root / "reports" / (
            f"{grid.PROTOCOL_ID}_N254_r1_unit_test_never_written.json"
        )
        protocol = grid.build_preregistration(self.root, planned)
        self.assertEqual(protocol["fixed_eta"], 1.1)
        self.assertEqual(protocol["fixed_alpha"], 20.0)
        self.assertFalse(protocol["scheduled_or_adaptive_eta"])
        self.assertFalse(protocol["energy_or_objective_backtracking_added"])
        self.assertFalse(protocol["barrier_fields_accessed_by_wrapper_or_launcher"])
        self.assertTrue(
            protocol["same_eta_alpha15_alpha_only_difference_verified"]
        )
        self.assertFalse(
            protocol["same_eta_alpha15_control_in_primary_success_gate"]
        )
        self.assertEqual(
            protocol["primary_comparators"],
            [
                "same_alpha_eta_one_candidate",
                "same_alpha_legacy",
                "official_stage4_v4",
            ],
        )
        self.assertEqual(
            set(protocol["decision_inputs"]),
            {
                "same_alpha_eta_one_candidate_result",
                "same_alpha_eta_one_candidate_audit",
                "same_alpha_eta_one_candidate_command",
                "same_alpha_legacy_result",
                "same_alpha_legacy_audit",
                "same_alpha_legacy_command",
                "official_stage4_v4",
                "same_eta_alpha15_preregistration",
                "same_eta_alpha15_report",
                "same_eta_alpha15_result",
                "same_eta_alpha15_member_audit",
                "same_eta_alpha15_formal_ctp",
                "same_eta_alpha15_postrun_audit",
            },
        )
        self.assertEqual(
            protocol["registered_command_sha256"],
            member.compact_command_sha256(member.single_run_command(self.root)[0]),
        )
        grid.validate_preregistration(protocol, self.root)
        drifted = copy.deepcopy(protocol)
        drifted["fixed_alpha"] = 12.5
        with self.assertRaisesRegex(RuntimeError, "frozen configuration"):
            grid.validate_preregistration(drifted, self.root)

    def test_no_outputs_exclusive_writes_and_cli_separation(self):
        grid._assert_no_outputs(self.root)
        with tempfile.TemporaryDirectory() as temporary:
            fake_root = Path(temporary)
            paths = member.member_paths(fake_root)
            result_dir = paths["result_dir"]
            assert isinstance(result_dir, Path)
            result_dir.mkdir(parents=True)
            companion = result_dir / (
                f"ivm_{paths['label']}_central_state_ivm_cfgdeadbeef0000_"
                "budget254.audit.json"
            )
            companion.write_text("{}\n", encoding="utf-8")
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
