from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import run_ivm_v9_discrete_exact_candidate_budget_solver as candidate_budget
import run_ivm_v9_discrete_ffrbk_takeover_exact_noreparam_shim as runtime
import run_ivm_v9_discrete_ffrbk_takeover_selector_shim as selector
import run_ivm_v9_total_internal_nf_budget as total_budget
import run_stage4_ala_discrete_exact_noreparam_alpha_grid_member as member
import run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0 as helper


class ExactNoReparamAlphaGridMemberTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        with np.load(
            cls.root / "inputs/common_initial_path_equidistant.npz",
            allow_pickle=False,
        ) as saved:
            cls.initial_path = np.asarray(saved["initial_path"], dtype=float)
        assert member.legacy.array_sha256(cls.initial_path) == (
            member.COMMON_INITIAL_ARRAY_SHA256
        )

    def test_eight_alpha_commands_are_unique_and_exact(self):
        labels, hashes, cell_ids = set(), set(), set()
        for alpha in member.ALPHAS:
            command, label = member.single_run_command(self.root, alpha)
            options = helper.command_options(command)
            labels.add(label)
            hashes.add(member.compact_command_sha256(command))
            cell_ids.add(member.cell_id(alpha))
            self.assertEqual(
                Path(command[1]).name,
                "run_ivm_v9_discrete_ffrbk_takeover_exact_noreparam_shim.py",
            )
            self.assertEqual(options["--alpha"], f"{alpha:g}")
            self.assertEqual(options["--reparam-every"], "0")
            self.assertEqual(options["--discrete-residual-ratio"], "0.1")
            self.assertEqual(options["--discrete-inner-max-steps"], "5000")
            self.assertEqual(options["--max-step"], "1")
            self.assertEqual(options[selector.MODE_FLAG], "one_way_takeover")
            self.assertEqual(options[total_budget.TOTAL_INTERNAL_NF_FLAG], "5000")
        self.assertEqual((len(labels), len(hashes), len(cell_ids)), (8, 8, 8))

    def test_execution_host_override_changes_only_routing(self):
        alpha = 1.0e6
        default_host = member.assigned_host(alpha)
        override = "local" if default_host == "remote" else "remote"
        default_command, default_label = member.single_run_command(
            self.root, alpha
        )
        override_command, override_label = member.single_run_command(
            self.root, alpha, override
        )
        self.assertNotEqual(default_label, override_label)
        self.assertEqual(member.cell_id(alpha), member.cell_id(alpha, override))
        self.assertEqual(
            member.canonical_cell_spec(alpha),
            member.canonical_cell_spec(alpha, override),
        )
        self.assertIn(
            f"members/{override}",
            override_command[override_command.index("--output-dir") + 1],
        )
        self.assertNotEqual(default_command, override_command)

    def test_invalid_alpha_or_execution_host_fails_closed(self):
        with self.assertRaises(ValueError):
            member.single_run_command(self.root, 15.0)
        with self.assertRaises(ValueError):
            member.single_run_command(self.root, 1.0, "third-host")

    def _paths(self, count):
        paths = []
        for index in range(count):
            path = self.initial_path.copy()
            if index:
                path[1, 0] += index * 1.0e-4
            paths.append(path)
        return np.asarray(paths)

    def _base_arrays(self, alpha, interrupted):
        arrays = {key: np.array(0) for key in member.REQUIRED_RESULT_KEYS}
        if interrupted:
            # One exact solve ends at 4960 NF. The next solve pays two complete
            # bands, then its third candidate is refused. Runner publishes no
            # partial path; the terminal counter-only record stays at p1.
            counters = np.ones((3, 22), dtype=int)
            counters[1, 1:-1] = 248
            counters[2, 1:-1] = 250
            costs = np.asarray([20, 4960, 5000])
            calls = np.asarray([1, 248, 250])
            stored_paths = self._paths(2)
            stored_paths = np.asarray(
                [stored_paths[0], stored_paths[1], stored_paths[1]]
            )
            selections = np.full(247, "H0", dtype="U8")
            outers = np.zeros(247, dtype=int)
            steps = np.arange(247, dtype=int)
            histories = np.minimum(steps, 7)
            solve_stops = np.asarray(
                ["evaluated-relative-natural-residual"], dtype="U64"
            )
            solve_residuals = np.asarray([0.08])
            solve_endpoint_nf = np.asarray([4960])
            raw_paths = stored_paths[1:2].copy()
            published_paths = stored_paths[1:2].copy()
            solve_aligned = np.asarray([True])
            exact_mask = np.asarray([True, True, False])
            candidate_required = np.full(250, 20, dtype=int)
            candidate_remaining = np.concatenate(
                (np.arange(4980, 0, -20, dtype=int), np.asarray([0]))
            )
            candidate_stopped = np.zeros(250, dtype=bool)
            candidate_stopped[-1] = True
            candidate_joint_before = np.concatenate(
                (np.arange(247), np.asarray([0, 1, 2]))
            )
            outer_stop = "evaluation_budget"
            selector_fixed_calls = 2
            selector_aborted_calls = 1
            selector_aborted_selections = 3
            preflight_crops = 1
            preflight_zero = 0
            complete_band_stops = 1
            total_stop = "total_internal_joint_budget_exhausted"
        else:
            # The final path has the same NF as the previous exact endpoint but
            # a different coordinate. Path identity must keep it non-exact.
            counters = np.ones((4, 22), dtype=int)
            counters[1, 1] = 1
            counters[1, 2:-1] = 2
            counters[2, 1] = 2
            counters[2, 2:-1] = 3
            counters[3] = counters[2]
            costs = np.asarray([20, 39, 59, 59])
            calls = np.asarray([1, 2, 3, 3])
            stored_paths = self._paths(4)
            selections = np.asarray(["H0", "LBFGS"])
            outers = np.asarray([0, 1])
            steps = np.asarray([0, 0])
            histories = np.asarray([0, 0])
            solve_stops = np.asarray(
                [
                    "evaluated-relative-natural-residual",
                    "evaluated-absolute-natural-residual",
                ],
                dtype="U64",
            )
            solve_residuals = np.asarray([0.08, 0.0])
            solve_endpoint_nf = np.asarray([39, 59])
            raw_paths = stored_paths[1:3].copy()
            published_paths = stored_paths[1:3].copy()
            solve_aligned = np.asarray([True, True])
            exact_mask = np.asarray([True, True, True, False])
            candidate_required = np.asarray([19, 20])
            candidate_remaining = np.asarray([4980, 4961])
            candidate_stopped = np.asarray([False, False])
            candidate_joint_before = np.asarray([0, 0])
            outer_stop = "cached_coordinate_stall"
            selector_fixed_calls = 2
            selector_aborted_calls = 0
            selector_aborted_selections = 0
            preflight_crops = 0
            preflight_zero = 0
            complete_band_stops = 0
            total_stop = "runner_stopped_before_total_internal_joint_budget"

        consumed = int(costs[-1])
        remaining = 5000 - consumed
        complete_indices = np.flatnonzero(exact_mask)
        complete_nf = costs[exact_mask]
        last_complete_index = int(complete_indices[-1])
        last_complete_nf = int(costs[last_complete_index])
        terminal_complete = bool(exact_mask[-1])
        charged_tail = consumed - last_complete_nf
        partial_tail = bool(not terminal_complete and charged_tail > 0)
        raw_hashes = np.asarray(
            [member.legacy.array_sha256(path) for path in raw_paths]
        )
        published_hashes = np.asarray(
            [member.legacy.array_sha256(path) for path in published_paths]
        )
        h0_count = int(np.count_nonzero(selections == "H0"))
        lbfgs_count = int(np.count_nonzero(selections == "LBFGS"))
        arrays.update(
            {
                "active_alpha": np.full(costs.size, alpha),
                "active_reparam_every": np.zeros(costs.size, dtype=int),
                "alpha": np.array(alpha),
                "alpha_mode": np.array("fixed"),
                "calls_per_image": calls,
                "configuration_sha256": np.array("cfg"),
                "counter_vectors": counters,
                "discrete_all_returned_states_evaluated": np.array(True),
                "discrete_audit_global_relative_residual": solve_residuals,
                "discrete_audit_history_count": histories,
                "discrete_audit_outer": outers,
                "discrete_audit_selection": selections,
                "discrete_audit_step": steps,
                "discrete_audit_stop_reason": solve_stops,
                "discrete_ffrbk_selector_aborted_selection_count": np.array(selector_aborted_selections),
                "discrete_ffrbk_selector_aborted_solver_calls": np.array(selector_aborted_calls),
                "discrete_ffrbk_selector_apply_h0_inverse_preconditioner_none_calls": np.array(0),
                "discrete_ffrbk_selector_committed_selection_count": np.array(selections.size),
                "discrete_ffrbk_selector_energy_backtracking": np.array(False),
                "discrete_ffrbk_selector_extra_physical_evaluations": np.array(0),
                "discrete_ffrbk_selector_fixed_solver_calls": np.array(selector_fixed_calls),
                "discrete_ffrbk_selector_inner_max_steps": np.array(5000),
                "discrete_ffrbk_selector_latch_reset_count": np.array(selector_fixed_calls),
                "discrete_ffrbk_selector_mode": np.array("one_way_takeover"),
                "discrete_ffrbk_selector_one_way_latch_is_preference": np.array(True),
                "discrete_ffrbk_selector_residual_ratio": np.array(0.1),
                "discrete_ffrbk_selector_uncommitted_selection_count": np.array(0),
                "discrete_history": np.array(7),
                "discrete_h0_selection_count": np.array(h0_count),
                "discrete_implementation_id": np.array(
                    "base+" + candidate_budget.POLICY_ID
                ),
                "discrete_inner_max_steps": np.array(5000),
                "discrete_lbfgs_selection_count": np.array(lbfgs_count),
                "discrete_residual_ratio": np.array(0.1),
                "discrete_step_multiplier_aborted_selection_count": np.array(selector_aborted_selections),
                "discrete_step_multiplier_energy_backtracking": np.array(False),
                "discrete_step_multiplier_eta": np.array(1.1),
                "discrete_step_multiplier_extra_physical_evaluations": np.array(0),
                "discrete_step_multiplier_uncommitted_selection_count": np.array(0),
                "discrete_stop_mode": np.array("r08c10"),
                "exact_no_reparam_all_recorded_states_complete": np.array(bool(np.all(exact_mask))),
                "exact_no_reparam_candidate_budget_policy_id": np.array(candidate_budget.POLICY_ID),
                "exact_no_reparam_candidate_joint_steps_before": candidate_joint_before,
                "exact_no_reparam_candidate_remaining_internal_nf_before": candidate_remaining,
                "exact_no_reparam_candidate_required_internal_misses": candidate_required,
                "exact_no_reparam_candidate_stopped_before_query": candidate_stopped,
                "exact_no_reparam_charged_noncomplete_state_mask": ~exact_mask,
                "exact_no_reparam_completed_cap_stop_allowed": np.array(False),
                "exact_no_reparam_complete_band_preflight": np.array(True),
                "exact_no_reparam_complete_band_preflight_policy_id": np.array(runtime.COMPLETE_BAND_PREFLIGHT_POLICY_ID),
                "exact_no_reparam_complete_state_definition": np.array(
                    "initial_plus_published_outer_states_after_normalized_"
                    "residual_converged_frozen_solves"
                ),
                "exact_no_reparam_complete_state_indices": complete_indices,
                "exact_no_reparam_complete_state_mask": exact_mask,
                "exact_no_reparam_complete_state_nf": complete_nf,
                "exact_no_reparam_enabled": np.array(True),
                "exact_no_reparam_family": np.array(member.FAMILY),
                "exact_no_reparam_implementation_safety_inner_cap": np.array(5000),
                "exact_no_reparam_initial_nf": np.array(20),
                "exact_no_reparam_initial_path": self.initial_path,
                "exact_no_reparam_initial_path_sha256": np.array(member.COMMON_INITIAL_ARRAY_SHA256),
                "exact_no_reparam_inner_cap_required_nonbinding": np.array(True),
                "exact_no_reparam_operational_definition": np.array(runtime.OPERATIONAL_DEFINITION),
                "exact_no_reparam_residual_ratio": np.array(0.1),
                "exact_no_reparam_runtime_policy_id": np.array(runtime.POLICY_ID),
                "exact_no_reparam_scientific_maxstep_unbounded_required": np.array(True),
                "exact_no_reparam_solve_exact_endpoint_aligned_to_saved_state": solve_aligned,
                "exact_no_reparam_solve_exact_endpoint_nf": solve_endpoint_nf,
                "exact_no_reparam_solve_global_relative_residual": solve_residuals,
                "exact_no_reparam_solve_joint_steps": np.asarray([np.count_nonzero(outers == item) for item in dict.fromkeys(outers.tolist())]),
                "exact_no_reparam_solve_published_exact_endpoint_path": published_paths,
                "exact_no_reparam_solve_published_exact_endpoint_path_sha256": published_hashes,
                "exact_no_reparam_solve_raw_exact_endpoint_path": raw_paths,
                "exact_no_reparam_solve_raw_exact_endpoint_path_sha256": raw_hashes,
                "exact_no_reparam_solve_stop_reason": solve_stops,
                "exact_no_reparam_terminal_charged_tail_nf": np.array(charged_tail),
                "exact_no_reparam_terminal_record_is_complete": np.array(terminal_complete),
                "frozen_subproblem": np.array("discrete_b040"),
                "geometric_reparam_joint_evaluations": np.array(0),
                "i_perp": np.linspace(4.0, 3.0, costs.size),
                "initial_path_sha256": np.array(member.COMMON_INITIAL_ARRAY_SHA256),
                "max_step": np.array(1.0),
                "plbfgs_kkt_atol": np.array(0.0),
                "plbfgs_kkt_rtol": np.array(0.1),
                "reparam_attempted": np.zeros(costs.size, dtype=bool),
                "reparam_committed": np.zeros(costs.size, dtype=bool),
                "reparam_direct_joint_evaluations": np.zeros(costs.size, dtype=int),
                "reparam_every": np.array(0),
                "reparam_every_schedule": np.array(""),
                "reparam_mode": np.array("state_curve"),
                "reparam_recovery_joint_evaluations": np.zeros(costs.size, dtype=int),
                "state_reparam_accepts": np.array(0),
                "state_reparam_attempts": np.array(0),
                "state_reparam_recovery_joint_evaluations": np.array(0),
                "stop_complete_iperp": np.array(-1.0),
                "stop_iperp": np.array(-1.0),
                "stop_reason": np.array(outer_stop),
                "stored_path_calls": calls,
                "stored_paths": stored_paths,
                "total_internal_joint_evaluations": costs,
                "total_internal_nf_budget": np.array(5000),
                "total_internal_nf_budget_blocked_image_indices": np.asarray([], dtype=int),
                "total_internal_nf_budget_blocked_miss_count": np.array(0),
                "total_internal_nf_budget_exhausted": np.array(consumed == 5000),
                "total_internal_nf_budget_exhaustion_observed_at_cache_miss": np.array(False),
                "total_internal_nf_budget_no_overshoot": np.array(True),
                "total_internal_nf_budget_policy_id": np.array(total_budget.POLICY_ID),
                "total_internal_nf_budget_stop_reason": np.array(total_stop),
                "total_internal_nf_complete_band_budget_stop_count": np.array(complete_band_stops),
                "total_internal_nf_complete_state_definition": np.array(total_budget.COMPLETENESS_KEYED_MASK),
                "total_internal_nf_complete_state_mask_key": np.array(runtime.EXACT_COMPLETE_MASK_KEY),
                "total_internal_nf_consumed": np.array(consumed),
                "total_internal_nf_endpoint_evaluations_excluded": np.array(True),
                "total_internal_nf_full_band_preflight_calls": np.array(candidate_required.size),
                "total_internal_nf_full_band_preflight_crops": np.array(preflight_crops),
                "total_internal_nf_full_band_preflight_zero_step_stops": np.array(preflight_zero),
                "total_internal_nf_last_complete_state_index": np.array(last_complete_index),
                "total_internal_nf_last_complete_state_nf": np.array(last_complete_nf),
                "total_internal_nf_remaining": np.array(remaining),
                "total_internal_nf_terminal_partial_tail_present": np.array(partial_tail),
                "total_internal_nf_terminal_partial_tail_quarantined": np.array(partial_tail),
                "total_internal_nf_terminal_record_is_complete_outer_state": np.array(terminal_complete),
                "total_internal_nf_terminal_uncommitted_charged_tail_nf": np.array(charged_tail),
            }
        )
        return arrays

    @staticmethod
    def _write_result(directory, arrays):
        result = directory / "result.npz"
        np.savez(result, **arrays)
        consumed = int(np.asarray(arrays["total_internal_nf_consumed"]).item())
        counters = np.asarray(arrays["counter_vectors"], dtype=int)
        audit = {
            "consistent": True,
            "backend_raw_joint_evaluations": consumed + 2,
            "sum_image_joint_evaluations": consumed + 2,
            "per_image_joint_evaluations": [1]
            + counters[-1, 1:-1].tolist()
            + [1],
            "state_reparam_attempts": 0,
            "state_reparam_accepts": 0,
            "state_reparam_recovery_joint_evaluations": 0,
            "geometric_reparam_joint_evaluations": 0,
            "discrete_all_returned_states_evaluated": True,
        }
        result.with_suffix(".audit.json").write_text(
            json.dumps(audit), encoding="utf-8"
        )
        return result

    def test_validator_accepts_asymmetric_hits_absolute_zero_and_nf_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self._write_result(
                Path(temporary), self._base_arrays(1.0, False)
            )
            summary = member.validate_result(result, 1.0, "remote")
            self.assertEqual(summary["final_optimization_NF"], 59)
            self.assertEqual(summary["exact_complete_state_count"], 3)
            self.assertFalse(summary["all_recorded_states_exact_complete"])
            self.assertEqual(summary["absolute_residual_stop_count"], 1)
            self.assertEqual(
                summary["discrete_inner_step_count_including_cache_hits"], 2
            )
            self.assertEqual(summary["direction_proposal_count"], 2)

    def test_validator_quarantines_paid_interrupted_solve_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self._write_result(
                Path(temporary), self._base_arrays(1.0, True)
            )
            summary = member.validate_result(result, 1.0)
            self.assertEqual(summary["final_optimization_NF"], 5000)
            self.assertEqual(summary["last_exact_complete_NF"], 4960)
            self.assertEqual(summary["terminal_charged_tail_NF"], 40)
            self.assertFalse(summary["terminal_record_is_exact_complete"])
            self.assertEqual(summary["selector_aborted_solver_calls_at_budget"], 1)
            self.assertEqual(
                summary["discrete_inner_step_count_including_cache_hits"], 249
            )
            self.assertEqual(summary["direction_proposal_count"], 250)

    def test_validator_rejects_inner_safety_cap_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            arrays = self._base_arrays(1.0, False)
            arrays["discrete_audit_stop_reason"] = np.asarray(
                ["evaluated-inner-max-steps", "evaluated-absolute-natural-residual"]
            )
            arrays["exact_no_reparam_solve_stop_reason"] = arrays[
                "discrete_audit_stop_reason"
            ]
            result = self._write_result(Path(temporary), arrays)
            with self.assertRaises(RuntimeError):
                member.validate_result(result, 1.0)


if __name__ == "__main__":
    unittest.main()
