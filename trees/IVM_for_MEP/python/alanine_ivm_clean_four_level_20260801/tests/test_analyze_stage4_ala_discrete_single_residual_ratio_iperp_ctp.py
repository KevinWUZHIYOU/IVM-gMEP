from __future__ import annotations

import json
from pathlib import Path
import unittest

import analyze_stage4_ala_discrete_single_residual_ratio_iperp_ctp as analysis


class ResidualRatioIperpCtpAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.report = cls.root / "reports" / (
            "stage4_ala_discrete_single_residual_ratio_ablation_v1_"
            "N254_r1_1787253500284697000.json"
        )

    def test_npz_allowlists_exclude_barrier(self):
        for name, keys in analysis.NPZ_ACCESS_ALLOWLISTS.items():
            self.assertFalse(
                any("barrier" in key.lower() or "energy" in key.lower() for key in keys),
                name,
            )

    def test_cost_to_precision_is_actual_nf_without_interpolation(self):
        points = [
            {"NF": 10, "i_perp": 1.0},
            {"NF": 40, "i_perp": 0.2},
            {"NF": 90, "i_perp": 0.05},
        ]
        self.assertEqual(analysis.base.cost_to_precision(points, 0.2), 40)
        self.assertEqual(analysis.base.cost_to_precision(points, 0.1), 90)
        self.assertIsNone(analysis.base.cost_to_precision(points, 0.01))

    def test_full_curve_dominance_requires_tightest_precision(self):
        comparator = [
            {"NF": 20, "i_perp": 0.5},
            {"NF": 100, "i_perp": 0.1},
        ]
        candidate = [
            {"NF": 10, "i_perp": 0.5},
            {"NF": 90, "i_perp": 0.11},
        ]
        outcome = analysis.base.compare_curves(candidate, comparator, "control")
        self.assertFalse(outcome["candidate_does_not_lose_tightest_precision"])
        self.assertFalse(outcome["full_curve_dominates"])

    def test_registered_multiwinner_tiebreak_and_no_winner(self):
        def row(ratio, official, legacy, joint=True):
            def vector(values):
                return {
                    "candidate_minus_comparator_NF_tight_to_loose": list(values),
                    "candidate_unreachable_tight_to_loose": [
                        value is None for value in values
                    ],
                }

            return {
                "residual_ratio": ratio,
                "joint_full_curve_dominates": joint,
                "official_deficit_vector": vector(official),
                "legacy_deficit_vector": vector(legacy),
            }

        decision = analysis.decide_primary_winner(
            [
                row(0.9, (-5, -2), (-1, 0)),
                row(0.5, (-4, -9), (-20, -20)),
            ]
        )
        self.assertEqual(decision["selected_primary_winner_ratio"], 0.9)
        self.assertTrue(decision["winner_claim_made"])
        self.assertTrue(decision["barrier_unembargo_allowed"])

        none = analysis.decide_primary_winner(
            [row(0.9, (), (), False), row(0.5, (), (), False)]
        )
        self.assertIsNone(none["selected_primary_winner_ratio"])
        self.assertFalse(none["winner_claim_made"])
        self.assertFalse(none["ratio_lock_allowed"])
        self.assertFalse(none["barrier_unembargo_allowed"])

    def test_unreachable_deficit_serializes_as_null_with_explicit_flag(self):
        vector = analysis.deficit_vector(
            [{"NF": 20, "i_perp": 0.2}],
            [{"NF": 10, "i_perp": 0.1}],
        )
        self.assertEqual(
            vector["candidate_minus_comparator_NF_tight_to_loose"], [None]
        )
        self.assertEqual(vector["candidate_unreachable_tight_to_loose"], [True])
        self.assertNotIn("ordering_tuple", vector)
        json.dumps(vector, allow_nan=False)

    def test_complete_formal_chain_is_fail_closed_no_winner(self):
        payload = analysis.analyze(self.report)
        # This is the exact strict serialization contract used by the CLI.
        json.dumps(payload, allow_nan=False)
        self.assertEqual(payload["merged_residual_ratio_grid"], [0.9, 0.8, 0.5])
        self.assertEqual(payload["joint_full_curve_dominator_ratios"], [])
        self.assertIsNone(payload["selected_primary_winner_ratio"])
        self.assertEqual(
            payload["ratio_lock_status"], "not_locked_no_primary_dominator"
        )
        self.assertFalse(payload["winner_claim_made"])
        self.assertFalse(payload["ratio_lock_allowed"])
        self.assertFalse(payload["barrier_unembargo_allowed"])
        self.assertFalse(payload["barrier_data_accessed"])
        self.assertFalse(payload["barrier_computation_or_densification_performed"])
        self.assertFalse(payload["barrier_unembargoed"])
        self.assertTrue(payload["barrier_remains_embargoed"])
        self.assertTrue(payload["no_primary_dominator_rule_applied"])
        self.assertEqual(len(payload["formal_result_hashes"]), 4)
        self.assertEqual(len(payload["formal_member_audit_hashes"]), 4)
        self.assertEqual(len(payload["comparisons"]), 3)
        for comparison in payload["comparisons"]:
            self.assertFalse(comparison["joint_full_curve_dominates"])


if __name__ == "__main__":
    unittest.main()
