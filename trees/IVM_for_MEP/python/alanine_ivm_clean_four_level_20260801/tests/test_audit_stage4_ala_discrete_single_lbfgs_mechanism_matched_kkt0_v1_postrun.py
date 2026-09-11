#!/usr/bin/env python3
"""Read-only post-run audit for the incomplete matched-KKT0 v1 campaign.

The audit deliberately never indexes a physical energy or barrier array.  It
validates the saved Current0 control and the scalar-selector terminal
trajectory, including the distinction between the runner's genuine
``cached_coordinate_stall`` and the member wrapper's later shape-assumption
failure.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

import run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0 as member


REPORT_NAME = (
    "stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
    "N254_r1_1787260497802476000_postrun_mechanism_provenance_"
    "readonly_v1.json"
)
PREREG_NAME = (
    "stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
    "N254_r1_1787260497802476000_preregistered.json"
)
GRID_NAME = (
    "stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
    "N254_r1_1787260497802476000.json"
)


class MatchedKktZeroPostrunAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.report_path = cls.root / "reports" / REPORT_NAME
        cls.prereg_path = cls.root / "reports" / PREREG_NAME
        cls.grid_path = cls.root / "reports" / GRID_NAME
        cls.report = json.loads(cls.report_path.read_text(encoding="utf-8"))
        cls.prereg = json.loads(cls.prereg_path.read_text(encoding="utf-8"))
        cls.grid = json.loads(cls.grid_path.read_text(encoding="utf-8"))
        result_dir = (
            cls.root / "runs" / member.MEMBER_POLICY_VERSION / "members"
        )
        cls.current = next(result_dir.glob("*current_ffrbk_selector*.npz"))
        cls.scalar = next(result_dir.glob("*scalar_selector*.npz"))

    def test_frozen_hashes_and_grid_state(self) -> None:
        self.assertEqual(
            member._file_sha256(self.prereg_path),
            "04de96743da079b8b35247214a4ad3cf387157edb2134381cc08f95d3bab1f86",
        )
        self.assertEqual(
            member._file_sha256(self.grid_path),
            "c1506422a651ee664b8cdb08a19c124bbab760435130042c300b7e721e7fa02f",
        )
        self.assertEqual(self.grid["preregistration_sha256"], member._file_sha256(self.prereg_path))
        for item in self.prereg["run_source_manifest"].values():
            self.assertEqual(member._file_sha256(Path(item["path"])), item["sha256"])
        self.assertEqual(self.grid["status"], "incomplete")
        self.assertEqual([row["arm"] for row in self.grid["results"]], [
            "current_ffrbk_selector",
            "scalar_selector",
        ])
        self.assertEqual([row["returncode"] for row in self.grid["results"]], [0, 1])
        self.assertIn("alpha is not fixed throughout N254", self.grid["results"][1]["error"])
        self.assertEqual(self.report["p0_findings"], [])
        self.assertEqual(self.report["campaign_status"], "incomplete")

    def test_actual_commands_are_the_registered_commands(self) -> None:
        for arm in ("current_ffrbk_selector", "scalar_selector"):
            label = member.run_label(arm)
            actual_path = (
                self.root
                / "logs"
                / member.MEMBER_POLICY_VERSION
                / "members"
                / label
                / "command.json"
            )
            actual = json.loads(actual_path.read_text(encoding="utf-8"))
            registered = self.prereg["registered_commands"][arm]
            self.assertEqual(actual, registered["command"])
            self.assertEqual(
                member._compact_command_sha256(actual),
                registered["compact_sha256"],
            )
            launcher_path = (
                self.root
                / "logs"
                / member.MEMBER_POLICY_VERSION
                / "launcher"
                / label
                / "command.json"
            )
            launcher = json.loads(launcher_path.read_text(encoding="utf-8"))
            self.assertEqual(
                launcher,
                [
                    "/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python",
                    str(
                        (
                            self.root
                            / "src/run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0.py"
                        ).resolve()
                    ),
                    "--root",
                    str(self.root.resolve()),
                    "--arm",
                    arm,
                ],
            )
        direct_label = member.run_label("direct_lbfgs_only")
        self.assertFalse(any((self.root / "runs" / member.MEMBER_POLICY_VERSION / "members").glob(f"*{direct_label}*.npz")))
        self.assertFalse((self.root / "logs" / member.MEMBER_POLICY_VERSION / "launcher" / direct_label).exists())

    def test_current0_is_exact_on_all_registered_safe_fields(self) -> None:
        old = (self.root / member.OLD_CURRENT_RESULT).resolve()
        self.assertEqual(member._file_sha256(old), member.OLD_CURRENT_RESULT_SHA256)
        with np.load(self.current, allow_pickle=False) as current, np.load(
            old, allow_pickle=False
        ) as oracle:
            self.assertEqual(len(member.CURRENT0_PARITY_KEYS), 57)
            for key in sorted(member.CURRENT0_PARITY_KEYS):
                self.assertIn(key, current.files)
                self.assertIn(key, oracle.files)
                self.assertTrue(
                    member._equal_nan_exact(current[key], oracle[key]), key
                )
            self.assertEqual(np.asarray(current["plbfgs_kkt_atol"]).item(), 0.0)
            self.assertEqual(np.asarray(oracle["plbfgs_kkt_atol"]).item(), 1.0e-8)
        self.assertEqual(
            member._file_sha256(self.current),
            "eff9d585dfbbd966744a972f9fa5b5a7b3da11639d2079c333560e34ce98ee07",
        )
        current_report = self.report["current0"]
        self.assertTrue(current_report["old_safe_whole_run_exact_parity"])
        self.assertEqual(current_report["safe_parity_field_count"], 57)

    def test_scalar_mechanism_stop_accounting_and_reparameterization(self) -> None:
        with np.load(self.scalar, allow_pickle=False) as saved:
            self.assertEqual(np.asarray(saved["stop_reason"]).item(), "cached_coordinate_stall")
            active_alpha = np.asarray(saved["active_alpha"], dtype=float)
            cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
            calls = np.asarray(saved["calls_per_image"], dtype=int)
            counters = np.asarray(saved["counter_vectors"], dtype=int)
            self.assertEqual(active_alpha.shape, (58,))
            self.assertTrue(np.all(active_alpha == 15.0))
            self.assertTrue(np.array_equal(calls, np.arange(1, 59)))
            self.assertTrue(np.array_equal(cost, 20 * calls))
            self.assertEqual(int(cost[-1]), 1160)
            self.assertEqual(int(np.sum(counters[-1, 1:-1])), 1160)
            self.assertEqual(int(np.sum(counters[-1])), 1162)

            scalar_values = {
                "discrete_scalar_lbfgs_mode": "scalar_selector",
                "discrete_scalar_lbfgs_alpha": 15.0,
                "discrete_scalar_lbfgs_eta": 1.1,
                "discrete_scalar_lbfgs_history": 7,
                "discrete_scalar_lbfgs_stop_mode": "r08c10",
                "discrete_scalar_lbfgs_residual_ratio": 0.8,
                "discrete_scalar_lbfgs_inner_max_steps": 10,
                "discrete_scalar_lbfgs_kkt_atol": 0.0,
                "discrete_scalar_lbfgs_fixed_solver_call_count": 10,
                "discrete_scalar_lbfgs_runner_bank_argument_intercept_count": 10,
                "discrete_scalar_lbfgs_ffrbk_solve_normal_calls_inside_fixed_solver": 0,
                "discrete_scalar_lbfgs_ffrbk_solve_normal_calls_outside_fixed_solver": 40,
                "discrete_scalar_lbfgs_bank_instance_count": 1,
                "discrete_scalar_lbfgs_bank_build_count": 12,
                "discrete_scalar_lbfgs_aborted_solver_call_count": 0,
                "discrete_scalar_lbfgs_uncommitted_selection_count": 0,
                "discrete_scalar_lbfgs_extra_physical_evaluations": 0,
                "discrete_h0_selection_count": 34,
                "discrete_lbfgs_selection_count": 46,
            }
            for key, expected in scalar_values.items():
                self.assertEqual(np.asarray(saved[key]).item(), expected, key)
            true_flags = (
                "discrete_all_returned_states_evaluated",
                "discrete_scalar_lbfgs_selector_competition",
                "discrete_scalar_lbfgs_fixed_solver_preconditioner_is_none",
                "discrete_scalar_lbfgs_bank_build_zero_nf",
            )
            false_flags = (
                "discrete_scalar_lbfgs_absolute_kkt_stop_enabled",
                "discrete_scalar_lbfgs_direct_only",
                "discrete_scalar_lbfgs_ffrbk_applied_inside_fixed_solver",
                "discrete_scalar_lbfgs_energy_backtracking",
                "discrete_scalar_lbfgs_reparameterization_inside_subproblem",
            )
            self.assertTrue(all(bool(np.asarray(saved[key]).item()) for key in true_flags))
            self.assertFalse(any(bool(np.asarray(saved[key]).item()) for key in false_flags))
            self.assertTrue(np.all(np.asarray(saved["discrete_scalar_lbfgs_bank_init_raw_joint_evaluation_delta"], dtype=int) == 0))
            self.assertTrue(np.all(np.asarray(saved["discrete_scalar_lbfgs_bank_build_raw_joint_evaluation_delta"], dtype=int) == 0))
            self.assertTrue(np.all(np.asarray(saved["subproblem_line_search_backtracks"], dtype=int) == 0))

            selection = np.asarray(saved["discrete_audit_selection"], dtype="U8")
            audit_selection = np.asarray(saved["discrete_scalar_lbfgs_audit_selection"], dtype="U8")
            history = np.asarray(saved["discrete_scalar_lbfgs_audit_history_before"], dtype=int)
            gamma = np.asarray(saved["discrete_scalar_lbfgs_audit_initial_inverse_scale"], dtype=float)
            raw_gamma = np.asarray(saved["discrete_scalar_lbfgs_audit_raw_initial_inverse_scale"], dtype=float)
            sy = np.asarray(saved["discrete_scalar_lbfgs_audit_last_sy"], dtype=float)
            yy = np.asarray(saved["discrete_scalar_lbfgs_audit_last_yy"], dtype=float)
            self.assertTrue(np.array_equal(selection, audit_selection))
            self.assertEqual(selection.shape, (80,))
            empty = history == 0
            self.assertEqual(int(np.count_nonzero(empty)), 10)
            self.assertTrue(np.all(selection[empty] == "H0"))
            self.assertTrue(np.all(gamma[empty] == 1.0))
            self.assertTrue(np.array_equal(gamma[~empty], raw_gamma[~empty]))
            self.assertTrue(np.array_equal(raw_gamma[~empty], sy[~empty] / yy[~empty]))
            self.assertFalse(np.any(np.asarray(saved["discrete_scalar_lbfgs_audit_gamma_floor_applied"], dtype=bool)))
            self.assertFalse(np.any(np.asarray(saved["discrete_scalar_lbfgs_audit_direct_fallback"], dtype=bool)))

            base = np.asarray(saved["discrete_scalar_lbfgs_audit_base_max_atomic_displacement"], dtype=float)
            requested = np.asarray(saved["discrete_scalar_lbfgs_audit_requested_max_atomic_displacement"], dtype=float)
            applied = np.asarray(saved["discrete_scalar_lbfgs_audit_applied_max_atomic_displacement"], dtype=float)
            dynamic = np.asarray(saved["discrete_scalar_lbfgs_audit_dynamic_max_step"], dtype=float)
            cap_hit = np.asarray(saved["discrete_scalar_lbfgs_audit_cap_hit"], dtype=bool)
            self.assertTrue(np.allclose(requested, 1.1 * base, rtol=1.0e-13, atol=1.0e-14))
            self.assertTrue(np.all(applied <= dynamic * (1.0 + 1.0e-12)))
            self.assertTrue(np.array_equal(cap_hit, requested > dynamic))
            self.assertEqual(int(np.count_nonzero(cap_hit)), 5)

            outer = np.asarray(saved["discrete_audit_outer"], dtype=int)
            step = np.asarray(saved["discrete_audit_step"], dtype=int)
            ordered = list(dict.fromkeys(outer.tolist()))
            self.assertEqual([int(np.count_nonzero(outer == value)) for value in ordered], [3, 4, 7, 8, 8, 10, 10, 10, 10, 10])
            reasons = np.asarray(saved["discrete_audit_stop_reason"], dtype="U64")
            relative = np.asarray(saved["discrete_audit_global_relative_residual"], dtype=float)
            self.assertTrue(np.all(reasons[:5] == "evaluated-relative-natural-residual"))
            self.assertTrue(np.all(relative[:5] <= 0.8))
            self.assertTrue(np.all(reasons[5:] == "evaluated-inner-max-steps"))
            for outer_value in ordered:
                local = step[outer == outer_value]
                self.assertTrue(np.array_equal(local, np.arange(local.size)))

            self.assertEqual(np.asarray(saved["reparam_mode"]).item(), "state_curve")
            self.assertEqual(np.asarray(saved["state_curve_direct_objective_acceptance"]).item(), "per_image")
            self.assertFalse(bool(np.asarray(saved["state_curve_projection_uses_spline"]).item()))
            self.assertEqual(np.asarray(saved["state_curve_projection_openmm_calls"]).item(), 0)
            self.assertEqual(np.asarray(saved["state_reparam_attempts"]).item(), 6)
            self.assertEqual(np.asarray(saved["state_reparam_accepts"]).item(), 5)
            self.assertEqual(np.asarray(saved["state_reparam_recovery_joint_evaluations"]).item(), 20)

        self.assertEqual(
            member._file_sha256(self.scalar),
            "ef5b54cd3127c86abf82ad934bb932f89e17e3008037fbd1c38a2570d868aea4",
        )

    def test_mesh_and_ctp_u_semantics_use_only_observed_records(self) -> None:
        with np.load(self.scalar, allow_pickle=False) as saved:
            cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
            iperp = np.asarray(saved["i_perp"], dtype=float)
            complete = np.asarray(saved["reparam_committed"], dtype=bool).copy()
            complete[0] = True
            cv = np.asarray(saved["segment_length_cv"], dtype=float)
            ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
            eligible = complete & (cv <= 0.0065) & (ratio <= 1.03)
            self.assertEqual(cost[eligible].tolist(), [20, 100, 200, 560, 740, 960])
            self.assertEqual(float(np.min(iperp[eligible])), 0.14394805708615313)
            anchors = [1.0, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.00015]
            observed = {}
            for threshold in anchors:
                reached = eligible & (iperp <= threshold)
                observed[str(threshold)] = int(np.min(cost[reached])) if np.any(reached) else "U"
        self.assertEqual(observed, self.report["scalar_selector"]["reporting_anchor_ctp_preview"])
        self.assertEqual(observed["1.0"], 560)
        self.assertTrue(all(observed[str(value)] == "U" for value in anchors[1:]))
        self.assertEqual(self.report["ctp_interpretation"]["unobserved_direct_arm"], "NA_not_launched_not_U")
        self.assertEqual(self.report["ctp_interpretation"]["scalar_unreached_threshold"], "U_and_a_loss_under_preregistration")

    def test_no_physical_energy_or_barrier_array_was_used(self) -> None:
        self.assertFalse(self.report["physical_energy_arrays_accessed"])
        self.assertFalse(self.report["barrier_arrays_accessed"])
        self.assertFalse(self.report["densified_barrier_computation_performed"])
        accessed = set(self.report["npz_safe_fields_accessed"])
        forbidden = {
            "barrier_kcal",
            "barrier_metric_for_figures",
            "energy_request_vectors",
            "optimization_barrier_is_sparse",
        }
        self.assertTrue(accessed.isdisjoint(forbidden))


if __name__ == "__main__":
    unittest.main()
