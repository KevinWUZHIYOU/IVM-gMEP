from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import audit_chignolin_ir_ivm_results as audit


def _counter_vectors(cost: np.ndarray) -> np.ndarray:
    counters = np.ones((cost.size, 22), dtype=np.int64)
    for row, internal_nf in enumerate(cost):
        quotient, remainder = divmod(int(internal_nf), 20)
        counters[row, 1:-1] = quotient
        counters[row, 1 : 1 + remainder] += 1
    return counters


class AuditChignolinIRIVMResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_pair(
        self,
        name: str = "result",
        npz_updates: Optional[Dict[str, Any]] = None,
        companion_updates: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Path]:
        cost = np.array([20, 40, 60, 80, 90], dtype=np.int64)
        counters = _counter_vectors(cost)
        calls = np.max(counters[:, 1:-1], axis=1)
        committed = np.array([False, False, True, False, False])
        direct_nf = np.array([0, 20, 20, 0, 10], dtype=np.int64)
        recovery_nf = np.array([0, 0, 5, 0, 0], dtype=np.int64)
        stored_paths = np.arange(5 * 22 * 3, dtype=float).reshape(5, 22, 3)
        implementation_id = "synthetic-selector-plus-eta-implementation"
        data: Dict[str, Any] = {
            "active_alpha": np.full(5, 10.0),
            "active_reparam_every": np.ones(5, dtype=np.int64),
            "alpha": np.array(10.0),
            "alpha_mode": np.array("fixed"),
            "budget": np.array(100),
            "calls_per_image": calls,
            "configuration_sha256": np.array("a" * 64),
            "counter_vectors": counters,
            "discrete_all_returned_states_evaluated": np.array(True),
            "discrete_ffrbk_selector_alpha": np.array(10.0),
            "discrete_ffrbk_selector_final_implementation_id": np.array(
                implementation_id
            ),
            "discrete_ffrbk_selector_history": np.array(7),
            "discrete_ffrbk_selector_inner_max_steps": np.array(5),
            "discrete_ffrbk_selector_mode": np.array("one_way_takeover"),
            "discrete_ffrbk_selector_post_select_eta_composed": np.array(True),
            "discrete_ffrbk_selector_reparameterization_inside_subproblem": np.array(
                False
            ),
            "discrete_ffrbk_selector_residual_ratio": np.array(0.5),
            "discrete_ffrbk_selector_eta": np.array(1.1),
            "discrete_history": np.array(7),
            "discrete_implementation_id": np.array(implementation_id),
            "discrete_inner_max_steps": np.array(5),
            "discrete_residual_ratio": np.array(0.5),
            "discrete_step_multiplier_eta": np.array(1.1),
            "discrete_step_multiplier_implementation_id": np.array(
                implementation_id
            ),
            "frozen_subproblem": np.array("discrete_b040"),
            # Records 1, 3, and 4 are deliberately lower than the threshold,
            # but they are not committed complete path states.
            "i_perp": np.array([1.0, 0.10, 0.20, 0.05, 0.01]),
            "initial_path": stored_paths[0],
            "max_step": np.array(1.0),
            "reparam_attempted": np.array([False, True, True, False, True]),
            "reparam_committed": committed,
            "reparam_direct_joint_evaluations": direct_nf,
            "reparam_every": np.array(1),
            "reparam_mode": np.array("state_curve"),
            "reparam_recovery_joint_evaluations": recovery_nf,
            "run_label": np.array("chig_irivm_a1e1_r05_m05_inf100_r1"),
            "state_curve_direct_objective_acceptance": np.array("per_image"),
            "state_curve_direct_solve": np.array(True),
            "state_reparam_accepts": np.array(1),
            "state_reparam_attempts": np.array(3),
            "state_reparam_recovery_joint_evaluations": np.array(5),
            "stored_path_calls": calls,
            "stored_paths": stored_paths,
            "total_internal_joint_evaluations": cost,
            "total_internal_nf_budget": np.array(100),
            "total_internal_nf_budget_blocked_miss_count": np.array(0),
            "total_internal_nf_budget_exhausted": np.array(False),
            "total_internal_nf_budget_no_overshoot": np.array(True),
            "total_internal_nf_complete_state_definition": np.array(
                "initial_or_reparam_committed"
            ),
            "total_internal_nf_consumed": np.array(90),
            "total_internal_nf_endpoint_evaluations_excluded": np.array(True),
            "total_internal_nf_last_complete_state_index": np.array(2),
            "total_internal_nf_last_complete_state_nf": np.array(60),
            "total_internal_nf_remaining": np.array(10),
            "total_internal_nf_terminal_partial_tail_present": np.array(True),
            "total_internal_nf_terminal_partial_tail_quarantined": np.array(True),
            "total_internal_nf_terminal_record_is_complete_outer_state": np.array(
                False
            ),
            "total_internal_nf_terminal_uncommitted_charged_tail_nf": np.array(30),
        }
        if npz_updates:
            data.update(npz_updates)

        npz_path = self.root / (name + ".npz")
        np.savez(npz_path, **data)
        final_counters = np.asarray(data["counter_vectors"])[-1]
        companion: Dict[str, Any] = {
            "backend_label": "ivm_optimization",
            "backend_raw_joint_evaluations": int(np.sum(final_counters)),
            "consistent": True,
            "discrete_history": int(np.asarray(data["discrete_history"]).item()),
            "discrete_implementation_id": str(
                np.asarray(data["discrete_implementation_id"]).item()
            ),
            "discrete_inner_max_steps": int(
                np.asarray(data["discrete_inner_max_steps"]).item()
            ),
            "discrete_residual_ratio": float(
                np.asarray(data["discrete_residual_ratio"]).item()
            ),
            "per_image_budget_blocked_misses": [0] * 22,
            "per_image_joint_evaluations": final_counters.tolist(),
            "reparam_mode": str(np.asarray(data["reparam_mode"]).item()),
            "result_file": str(npz_path.resolve()),
            "state_reparam_accepts": int(
                np.asarray(data["state_reparam_accepts"]).item()
            ),
            "state_reparam_attempts": int(
                np.asarray(data["state_reparam_attempts"]).item()
            ),
            "state_reparam_recovery_joint_evaluations": int(
                np.asarray(data["state_reparam_recovery_joint_evaluations"]).item()
            ),
            "sum_image_joint_evaluations": int(np.sum(final_counters)),
        }
        if companion_updates:
            companion.update(companion_updates)
        companion_path = self.root / (name + ".audit.json")
        companion_path.write_text(
            json.dumps(companion, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return npz_path, companion_path

    def test_valid_pair_uses_only_initial_or_committed_states(self) -> None:
        npz_path, companion_path = self._make_pair()
        result = audit.audit_result(npz_path, companion_path)

        self.assertEqual(result["terminal_total_nf"], 92)
        self.assertEqual(result["usable_state_indices"], [0, 2])
        self.assertEqual(result["first_threshold_state_index"], 2)
        self.assertEqual(result["first_threshold_state_kind"], "reparam_committed")
        self.assertEqual(result["first_threshold_total_nf"], 62)
        self.assertEqual(result["last_complete_state_index"], 2)
        self.assertEqual(result["last_complete_total_nf"], 62)
        self.assertEqual(result["terminal_partial_tail_internal_nf"], 30)
        self.assertTrue(result["terminal_partial_tail_quarantined"])
        self.assertEqual(result["reparam_attempt_count"], 3)
        self.assertEqual(result["recovery_event_count"], 1)
        self.assertEqual(result["recovery_joint_evaluations"], 5)
        self.assertFalse(result["recovery_event_count_is_lower_bound"])
        self.assertEqual(
            result["recovery_joint_evaluations_assigned_to_saved_record"], 5
        )
        self.assertEqual(
            result["recovery_joint_evaluations_unassigned_to_saved_record"], 0
        )
        self.assertEqual(
            result["recovery_saved_record_gap_evidence"],
            "exact_saved_record_sum",
        )

    def test_compacted_rollback_closure_gap_requires_counter_evidence(self) -> None:
        npz_path, companion_path = self._make_pair(
            name="compacted_closure",
            npz_updates={
                "reparam_attempted": np.array([False, True, True, False, True]),
                "reparam_committed": np.array([False, False, False, False, True]),
                "reparam_direct_joint_evaluations": np.array(
                    [0, 20, 0, 0, 20], dtype=np.int64
                ),
                "reparam_recovery_joint_evaluations": np.array(
                    [0, 0, 0, 0, 10], dtype=np.int64
                ),
                "state_reparam_attempts": np.array(2),
                "state_reparam_accepts": np.array(1),
                "state_reparam_recovery_joint_evaluations": np.array(35),
                "total_internal_nf_last_complete_state_index": np.array(4),
                "total_internal_nf_last_complete_state_nf": np.array(90),
                "total_internal_nf_terminal_record_is_complete_outer_state": np.array(
                    True
                ),
                "total_internal_nf_terminal_uncommitted_charged_tail_nf": np.array(0),
                "total_internal_nf_terminal_partial_tail_present": np.array(False),
                "total_internal_nf_terminal_partial_tail_quarantined": np.array(False),
            },
        )

        result = audit.audit_result(npz_path, companion_path)

        # The scalar remains authoritative only because one completed attempt
        # has a missing closure annotation bounded by exact counter differences.
        self.assertEqual(result["terminal_total_nf"], 92)
        self.assertEqual(result["last_complete_total_nf"], 92)
        self.assertEqual(result["recovery_joint_evaluations"], 35)
        self.assertEqual(
            result["recovery_joint_evaluations_assigned_to_saved_record"], 10
        )
        self.assertEqual(
            result["recovery_joint_evaluations_unassigned_to_saved_record"], 25
        )
        self.assertEqual(result["recovery_saved_record_missing_closure_count"], 1)
        self.assertEqual(result["recovery_saved_record_gap_counter_lower_bound"], 20)
        self.assertEqual(result["recovery_saved_record_gap_counter_upper_bound"], 40)
        self.assertEqual(
            result["recovery_saved_record_gap_evidence"],
            "complete_attempt_closure_gap_with_counter_bounds",
        )
        self.assertTrue(result["recovery_event_count_is_lower_bound"])

    def test_unbounded_recovery_classification_gap_is_rejected(self) -> None:
        npz_path, companion_path = self._make_pair(
            name="unbounded_closure",
            npz_updates={
                "reparam_attempted": np.array([False, True, True, False, True]),
                "reparam_committed": np.array([False, False, False, False, True]),
                "reparam_direct_joint_evaluations": np.array(
                    [0, 20, 0, 0, 20], dtype=np.int64
                ),
                "reparam_recovery_joint_evaluations": np.array(
                    [0, 0, 0, 0, 10], dtype=np.int64
                ),
                "state_reparam_attempts": np.array(2),
                "state_reparam_accepts": np.array(1),
                "state_reparam_recovery_joint_evaluations": np.array(55),
                "total_internal_nf_last_complete_state_index": np.array(4),
                "total_internal_nf_last_complete_state_nf": np.array(90),
                "total_internal_nf_terminal_record_is_complete_outer_state": np.array(
                    True
                ),
                "total_internal_nf_terminal_uncommitted_charged_tail_nf": np.array(0),
                "total_internal_nf_terminal_partial_tail_present": np.array(False),
                "total_internal_nf_terminal_partial_tail_quarantined": np.array(False),
            },
        )
        with self.assertRaisesRegex(audit.AuditError, "saved counter intervals"):
            audit.audit_result(npz_path, companion_path)

    def test_recovery_scalar_smaller_than_saved_records_is_rejected(self) -> None:
        npz_path, companion_path = self._make_pair(
            name="recovery_scalar_too_small",
            npz_updates={
                "state_reparam_recovery_joint_evaluations": np.array(4),
            },
        )
        with self.assertRaisesRegex(audit.AuditError, "smaller than the sum"):
            audit.audit_result(npz_path, companion_path)

    def test_positive_gap_cannot_use_terminal_partial_attempt_as_evidence(self) -> None:
        npz_path, companion_path = self._make_pair(
            name="terminal_open_attempt",
            npz_updates={
                "state_reparam_recovery_joint_evaluations": np.array(6),
            },
        )
        with self.assertRaisesRegex(audit.AuditError, "terminal open attempt"):
            audit.audit_result(npz_path, companion_path)

    def test_backend_must_equal_sum_of_22_image_counters(self) -> None:
        npz_path, companion_path = self._make_pair(
            companion_updates={"backend_raw_joint_evaluations": 93}
        )
        with self.assertRaisesRegex(audit.AuditError, "sum of 22 image counters"):
            audit.audit_result(npz_path, companion_path)

    def test_partial_tail_must_be_quarantined(self) -> None:
        npz_path, companion_path = self._make_pair(
            npz_updates={
                "total_internal_nf_terminal_partial_tail_quarantined": np.array(False)
            }
        )
        with self.assertRaisesRegex(audit.AuditError, "partial_tail_quarantined"):
            audit.audit_result(npz_path, companion_path)

    def test_budget_overshoot_is_rejected(self) -> None:
        npz_path, companion_path = self._make_pair(
            npz_updates={
                "budget": np.array(80),
                "run_label": np.array("chig_irivm_a1e1_r05_m05_inf80_r1"),
                "total_internal_nf_budget": np.array(80),
                "total_internal_nf_remaining": np.array(-10),
            }
        )
        with self.assertRaisesRegex(audit.AuditError, "budget was overshot"):
            audit.audit_result(npz_path, companion_path)

    def test_state_curve_and_control_metadata_are_hard_gates(self) -> None:
        cases = (
            ({"reparam_mode": np.array("none")}, "reparam_mode"),
            ({"discrete_ffrbk_selector_mode": np.array("best_each_step")}, "selector_mode"),
            ({"discrete_ffrbk_selector_eta": np.array(1.0)}, "selector_eta"),
            ({"discrete_ffrbk_selector_residual_ratio": np.array(0.3)}, "residual_ratio"),
            ({"discrete_ffrbk_selector_inner_max_steps": np.array(8)}, "inner_max_steps"),
            ({"discrete_ffrbk_selector_alpha": np.array(1.0)}, "selector_alpha"),
        )
        for index, (updates, expected) in enumerate(cases):
            with self.subTest(expected):
                npz_path, companion_path = self._make_pair(
                    name="drift_{}".format(index), npz_updates=updates
                )
                with self.assertRaises(audit.AuditError):
                    audit.audit_result(npz_path, companion_path)

    def test_cli_accepts_multiple_explicit_pairs_and_writes_csv_json(self) -> None:
        first_npz, first_audit = self._make_pair(name="first")
        second_npz, second_audit = self._make_pair(
            name="second",
            npz_updates={
                "active_alpha": np.full(5, 1.0),
                "alpha": np.array(1.0),
                "discrete_ffrbk_selector_alpha": np.array(1.0),
                "run_label": np.array("chig_irivm_a1e0_r05_m05_inf100_r1"),
            },
        )
        csv_path = self.root / "summary.csv"
        json_path = self.root / "summary.json"

        audit.main(
            [
                "--pair",
                str(first_npz),
                str(first_audit),
                "--pair",
                str(second_npz),
                str(second_audit),
                "--output-csv",
                str(csv_path),
                "--output-json",
                str(json_path),
            ]
        )

        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        document = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["alpha"] for row in rows], ["1.0", "10.0"])
        self.assertEqual(document["result_count"], 2)
        self.assertFalse(document["dense_barrier_computed"])
        self.assertEqual(
            [item["first_threshold_total_nf"] for item in document["results"]],
            [62, 62],
        )


if __name__ == "__main__":
    unittest.main()
