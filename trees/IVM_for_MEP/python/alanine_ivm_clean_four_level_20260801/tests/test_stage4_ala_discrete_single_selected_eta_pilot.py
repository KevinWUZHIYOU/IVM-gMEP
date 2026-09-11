from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import run_ivm_v9_discrete_selected_eta_shim as eta_shim
import run_stage4_ala_discrete_single_selected_eta_pilot as member
import run_stage4_ala_discrete_single_selected_eta_pilot_grid as grid


class SelectedEtaPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def test_registered_grid_tags_and_fixed_controls(self):
        self.assertEqual(member.ETA_VALUES, (1.0, 1.25, 1.5, 2.0))
        self.assertEqual(member.ETA_VALUES, eta_shim.PILOT_ETA_VALUES)
        self.assertEqual(
            [member.eta_tag(value) for value in member.ETA_VALUES],
            ["eta1", "eta1p25", "eta1p5", "eta2"],
        )
        self.assertEqual(member.CANONICAL_ALPHA, 12.5)
        self.assertEqual(member.CANONICAL_RATIO, 0.8)
        self.assertEqual(member.CANONICAL_MAX_EVALUATED_STEPS, 10)
        self.assertEqual(member.CANONICAL_HISTORY, 7)
        self.assertEqual(member.CANONICAL_BUDGET, 254)
        with self.assertRaisesRegex(ValueError, "one of"):
            member.single_run_command(self.root, 1.75)

    def test_commands_are_exact_clones_except_runner_routing_and_eta(self):
        canonical = member.canonical_reference_command(self.root)
        for eta in member.ETA_VALUES:
            command, label = member.single_run_command(self.root, eta)
            member.assert_eta_only_scientific_change(
                command, canonical, eta, self.root
            )
            self.assertEqual(
                command[1],
                str(
                    (
                        self.root
                        / "src/run_ivm_v9_discrete_selected_eta_shim.py"
                    ).resolve()
                ),
            )
            self.assertEqual(command[-2], eta_shim.ETA_FLAG)
            self.assertEqual(command[-1], member.ETA_TEXT[eta])
            self.assertIn(member.eta_tag(eta), label)

            observed = list(command[:-2])
            observed[1] = canonical[1]
            for flag in member.ROUTING_OPTIONS:
                index = observed.index(flag) + 1
                observed[index] = canonical[canonical.index(flag) + 1]
            self.assertEqual(observed, canonical)

    def _synthetic_eta_result(self, eta: float) -> dict[str, np.ndarray]:
        oracle = self.root / member.ETA_ONE_ORACLE
        required = member.COMMON_RESULT_KEYS | member.DISCRETE_RESULT_KEYS
        with np.load(oracle, allow_pickle=False) as saved:
            payload = {key: np.array(saved[key], copy=True) for key in required}

        implementation_id = eta_shim.shim_implementation_id(
            member.coarse_single.EXPECTED_IMPLEMENTATION_ID, eta
        )
        payload["discrete_implementation_id"] = np.asarray(implementation_id)
        selection = np.asarray(payload["discrete_audit_selection"], dtype="U8")
        size = selection.size
        base_max = np.full(size, 0.1)
        dynamic = np.full(size, 1.0)
        requested_max = eta * base_max
        applied_max = requested_max.copy()
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
                "discrete_step_multiplier_eta": np.asarray(eta),
                "discrete_step_multiplier_application_stage": np.asarray(
                    eta_shim.APPLICATION_STAGE
                ),
                "discrete_step_multiplier_selector_isolated": np.asarray(True),
                "discrete_step_multiplier_dynamic_cap_reapplied": np.asarray(
                    True
                ),
                "discrete_step_multiplier_energy_backtracking": np.asarray(False),
                "discrete_step_multiplier_extra_physical_evaluations": np.asarray(
                    0
                ),
                "discrete_step_multiplier_audit_selection": selection.copy(),
                "discrete_step_multiplier_audit_requested": np.full(size, eta),
                "discrete_step_multiplier_audit_applied": np.full(size, eta),
                "discrete_step_multiplier_audit_cap_hit": np.zeros(
                    size, dtype=bool
                ),
                "discrete_step_multiplier_audit_dynamic_max_step": dynamic,
                "discrete_step_multiplier_audit_base_max_atomic_displacement": (
                    base_max
                ),
                "discrete_step_multiplier_audit_requested_max_atomic_displacement": (
                    requested_max
                ),
                "discrete_step_multiplier_audit_applied_max_atomic_displacement": (
                    applied_max
                ),
                "discrete_step_multiplier_uncommitted_selection_count": np.asarray(
                    0
                ),
                "discrete_step_multiplier_aborted_selection_count": np.asarray(0),
                # Object dtype would fail under allow_pickle=False if the
                # validator indexed it.  Its presence audits the embargo.
                "barrier_kcal": np.asarray(["embargoed"], dtype=object),
            }
        )
        return payload

    def test_eta_one_validator_enforces_whole_run_parity_without_barrier(self):
        payload = self._synthetic_eta_result(1.0)
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "eta1.npz"
            np.savez(result, **payload)
            summary = member._validate_result_iperp_only(
                self.root, result, 1.0
            )
        self.assertTrue(summary["eta_one_whole_run_parity_or_not_applicable"])
        self.assertEqual(
            set(summary["eta_one_parity_keys"]), member.ETA_ONE_PARITY_KEYS
        )
        self.assertNotIn("barrier_kcal", summary["result_fields_accessed"])
        self.assertEqual(summary["eta_cap_hit_count"], 0)

    def test_eta_two_validator_accepts_post_select_recap_metadata(self):
        payload = self._synthetic_eta_result(2.0)
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "eta2.npz"
            np.savez(result, **payload)
            summary = member._validate_result_iperp_only(
                self.root, result, 2.0
            )
        self.assertFalse(summary["eta_one_whole_run_parity_or_not_applicable"])
        self.assertEqual(summary["eta_applied_multiplier_min"], 2.0)
        self.assertEqual(summary["eta_applied_multiplier_max"], 2.0)

    def test_validator_rejects_energy_backtracking_and_selector_drift(self):
        for key, replacement, message in (
            (
                "discrete_step_multiplier_energy_backtracking",
                np.asarray(True),
                "forbidden energy backtracking",
            ),
            (
                "discrete_step_multiplier_audit_selection",
                np.asarray([], dtype="U8"),
                "do not align",
            ),
        ):
            payload = self._synthetic_eta_result(1.25)
            payload[key] = replacement
            with tempfile.TemporaryDirectory() as temporary:
                result = Path(temporary) / "invalid.npz"
                np.savez(result, **payload)
                with self.assertRaisesRegex(RuntimeError, message):
                    member._validate_result_iperp_only(
                        self.root, result, 1.25
                    )

    def test_every_result_allowlist_excludes_barrier_fields(self):
        for keys in (
            member.COMMON_RESULT_KEYS,
            member.DISCRETE_RESULT_KEYS,
            member.ETA_RESULT_KEYS,
            member.ETA_ONE_PARITY_KEYS,
        ):
            self.assertFalse(any("barrier" in key.lower() for key in keys), keys)

    def test_preregistration_locks_inputs_sources_commands_and_no_winner_rule(self):
        planned = self.root / "reports" / (
            f"{grid.PROTOCOL_ID}_N254_r1_unit_test_never_written.json"
        )
        protocol = grid.build_preregistration(self.root, planned)
        self.assertEqual(protocol["eta_grid"], [1.0, 1.25, 1.5, 2.0])
        self.assertTrue(protocol["same_build_eta_one_control_is_newly_run"])
        self.assertTrue(protocol["eta_one_whole_run_parity_gate_required"])
        self.assertFalse(protocol["energy_or_objective_backtracking_added"])
        self.assertFalse(protocol["barrier_fields_accessed_by_wrapper_or_launcher"])
        self.assertEqual(
            set(protocol["registered_command_hashes"]),
            {"eta1", "eta1p25", "eta1p5", "eta2"},
        )
        self.assertEqual(
            set(protocol["decision_inputs"]),
            {
                "old_eta_one_identity_oracle_result",
                "old_eta_one_identity_oracle_audit",
                "old_eta_one_identity_oracle_command",
                "same_alpha_legacy_result",
                "same_alpha_legacy_audit",
                "same_alpha_legacy_command",
                "official_stage4_v4",
            },
        )
        self.assertFalse(
            protocol["no_primary_dominator_rule"]["eta_lock_allowed"]
        )
        self.assertFalse(
            protocol["no_primary_dominator_rule"]["barrier_unembargo_allowed"]
        )
        grid.validate_preregistration(protocol, self.root)

        drifted = copy.deepcopy(protocol)
        drifted["eta_grid"] = [1.0, 2.0]
        with self.assertRaisesRegex(RuntimeError, "eta grid"):
            grid.validate_preregistration(drifted, self.root)

    def test_no_outputs_exclusive_writes_and_cli_separation(self):
        grid._assert_no_outputs(self.root)
        paths = [member.member_paths(self.root, eta) for eta in member.ETA_VALUES]
        self.assertEqual(len({str(path["label"]) for path in paths}), 4)
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
