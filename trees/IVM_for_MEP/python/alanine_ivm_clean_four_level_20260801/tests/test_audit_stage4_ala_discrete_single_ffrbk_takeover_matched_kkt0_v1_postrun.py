from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import audit_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_postrun as audit


class FFrbkTakeoverPostrunAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.report = cls.root / audit.FORMAL_REPORT_RELATIVE

    def test_completed_campaign_passes_independent_audit(self) -> None:
        payload = audit.build_audit(self.root, self.report)
        self.assertEqual(payload["status"], "passed")
        self.assertEqual(payload["p0_findings"], [])
        self.assertFalse(payload["physical_energy_or_barrier_arrays_accessed"])
        self.assertFalse(payload["energy_or_objective_backtracking_used"])
        self.assertEqual(
            payload["provenance"]["registered_source_count"], 15
        )
        self.assertEqual(
            payload["provenance"]["registered_prior_artifact_count"], 15
        )
        self.assertTrue(
            payload["provenance"][
                "scientific_commands_exact_matched_current0_clones"
            ]
        )

        one_way = payload["arms"]["one_way_takeover"]
        forced = payload["arms"]["forced_operator_lbfgs_selection"]
        self.assertEqual(one_way["final_optimization_NF"], 3389)
        self.assertEqual(forced["final_optimization_NF"], 3389)
        self.assertEqual(one_way["relative_stop_count"], 22)
        self.assertEqual(forced["relative_stop_count"], 22)
        self.assertEqual(one_way["k10_stop_count"], 0)
        self.assertEqual(forced["k10_stop_count"], 0)
        self.assertEqual(one_way["ffrbk_apply_h0_inverse_call_count"], 120)
        self.assertEqual(forced["ffrbk_apply_h0_inverse_call_count"], 120)

        one_truth = one_way["selector_truth"]
        forced_truth = forced["selector_truth"]
        self.assertEqual(one_truth["h0_selection_count"], 22)
        self.assertEqual(one_truth["lbfgs_selection_count"], 49)
        self.assertEqual(one_truth["strict_win_trigger_count"], 19)
        self.assertEqual(one_truth["forced_lbfgs_count"], 30)
        self.assertEqual(forced_truth["h0_selection_count"], 0)
        self.assertEqual(forced_truth["lbfgs_selection_count"], 71)
        self.assertEqual(forced_truth["forced_lbfgs_count"], 71)
        for truth in (one_truth, forced_truth):
            failures = {
                key: value
                for key, value in truth.items()
                if key.endswith("_mismatch_count") and value
            }
            self.assertEqual(failures, {})
            self.assertEqual(truth["empty_history_tie_count"], 22)
            self.assertEqual(truth["h0_failover_count"], 0)

        comparison = payload["safe_scientific_trajectory_comparison"]
        self.assertEqual(
            comparison["safe_scientific_field_count"],
            len(audit.SAFE_SCIENTIFIC_TRAJECTORY_FIELDS),
        )
        self.assertTrue(comparison["all_safe_scientific_fields_exact"])
        self.assertTrue(comparison["stored_coordinate_trajectory_exact"])
        self.assertTrue(comparison["optimization_nf_trajectory_exact"])
        self.assertTrue(comparison["iperp_trajectory_exact"])
        self.assertEqual(comparison["mismatch_fields"], [])

    def test_truth_recomputation_rejects_latch_and_selection_drift(self) -> None:
        result = self.root / audit.RESULT_RELATIVE["one_way_takeover"]
        prefix = "discrete_ffrbk_selector_audit_"
        suffixes = (
            "selection",
            "solver_call",
            "selector_call",
            "latched_before",
            "latched_after",
            "strict_win_trigger",
            "forced_lbfgs",
            "empty_history_tie",
            "h0_failover",
            "h0_valid",
            "lbfgs_valid",
            "h0_capped_decrease",
            "lbfgs_capped_decrease",
        )
        with audit.SafeNpzReader(result) as reader:
            arrays = {suffix: reader.read(prefix + suffix).copy() for suffix in suffixes}
        arrays["latched_after"][4] = False
        with self.assertRaisesRegex(RuntimeError, "truth-table mismatch"):
            audit.recompute_selector_truth(arrays, "one_way_takeover")

    def test_safe_reader_fails_closed_on_energy_and_barrier_members(self) -> None:
        result = self.root / audit.RESULT_RELATIVE["one_way_takeover"]
        with audit.SafeNpzReader(result) as reader:
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                reader.read("barrier_kcal")
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                reader.read("energy_request_vectors")
            self.assertFalse(
                bool(reader.scalar("discrete_ffrbk_selector_energy_backtracking"))
            )
            self.assertEqual(
                reader.accessed,
                {"discrete_ffrbk_selector_energy_backtracking"},
            )

    def test_exact_comparison_detects_float_and_string_drift(self) -> None:
        self.assertTrue(
            audit._exact_equal(
                np.asarray([1.0, np.nan]), np.asarray([1.0, np.nan])
            )
        )
        self.assertFalse(
            audit._exact_equal(np.asarray([1.0]), np.asarray([1.0 + 1.0e-15]))
        )
        self.assertFalse(
            audit._exact_equal(np.asarray(["H0"]), np.asarray(["LBFGS"]))
        )

    def test_sidecar_write_is_exclusive_and_strict_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "sidecar.json"
            audit.write_exclusive_strict(target, {"status": "passed", "x": 1.0})
            self.assertEqual(json.loads(target.read_text())["status"], "passed")
            with self.assertRaises(FileExistsError):
                audit.write_exclusive_strict(target, {"status": "overwrite"})
            invalid = Path(temporary) / "invalid.json"
            with self.assertRaises(ValueError):
                audit.write_exclusive_strict(invalid, {"x": float("nan")})


if __name__ == "__main__":
    unittest.main()
