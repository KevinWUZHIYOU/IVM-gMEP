from __future__ import annotations

from pathlib import Path
import unittest

import analyze_stage4_ala_discrete_single_alpha_refinement_iperp_ctp as analysis


def point(nf: int, i_perp: float) -> dict[str, object]:
    return {"NF": nf, "i_perp": i_perp}


class RefinementIperpCtpAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.report = cls.root / "reports" / (
            "stage4_ala_discrete_single_alpha_refinement_matched_v1_"
            "N254_r1_1787252714040428000.json"
        )

    def test_ctp_is_exact_actual_nf_without_interpolation(self):
        points = [point(10, 1.0), point(30, 0.5), point(50, 0.2)]
        self.assertEqual(analysis.cost_to_precision(points, 0.75), 30)
        self.assertEqual(analysis.cost_to_precision(points, 0.5), 30)
        self.assertEqual(analysis.cost_to_precision(points, 0.3), 50)
        self.assertIsNone(analysis.cost_to_precision(points, 0.19))

    def test_full_curve_dominance_requires_weak_strict_and_tightest(self):
        comparator = [point(10, 1.0), point(30, 0.5), point(50, 0.2)]
        candidate = [point(10, 0.9), point(20, 0.5), point(40, 0.2)]
        result = analysis.compare_curves(candidate, comparator, "control")
        self.assertTrue(result["full_curve_dominates"])

        tied = analysis.compare_curves(comparator, comparator, "control")
        self.assertFalse(tied["full_curve_dominates"])

        loses_tightest = analysis.compare_curves(
            [point(5, 0.9), point(15, 0.3)], comparator, "control"
        )
        self.assertFalse(loses_tightest["candidate_does_not_lose_tightest_precision"])
        self.assertFalse(loses_tightest["full_curve_dominates"])

    def test_official_tail_order_prioritizes_unreachable_then_precision_then_ctp(self):
        official = [point(100, 0.1), point(200, 0.01), point(300, 0.001)]
        fewer_unreachable = [point(150, 0.1), point(250, 0.01)]
        more_unreachable = [point(50, 0.1)]
        _, fewer_key = analysis.official_tail_deficit(
            10.0, fewer_unreachable, official
        )
        _, more_key = analysis.official_tail_deficit(
            20.0, more_unreachable, official
        )
        self.assertLess(fewer_key, more_key)

        tighter = [point(500, 0.0005)]
        looser = [point(1, 0.001)]
        _, tighter_key = analysis.official_tail_deficit(10.0, tighter, official)
        _, looser_key = analysis.official_tail_deficit(20.0, looser, official)
        self.assertLess(tighter_key, looser_key)

        better_tight_ctp = [
            point(100, 0.1),
            point(200, 0.01),
            point(300, 0.001),
        ]
        better_loose_only = [
            point(90, 0.1),
            point(210, 0.01),
            point(300, 0.001),
        ]
        _, tight_key = analysis.official_tail_deficit(
            10.0, better_tight_ctp, official
        )
        _, loose_key = analysis.official_tail_deficit(
            20.0, better_loose_only, official
        )
        self.assertLess(tight_key, loose_key)

    def test_npz_access_allowlists_exclude_barrier_fields(self):
        allowlists = (
            analysis.COMMON_RESULT_KEYS,
            analysis.CANDIDATE_RESULT_KEYS,
            analysis.CONTROL_FORBIDDEN_RESULT_KEYS,
            analysis.OFFICIAL_ARCHIVE_KEYS,
            analysis.OFFICIAL_COMBINED_KEYS,
            analysis.OFFICIAL_COMPONENT_KEYS,
        )
        for allowlist in allowlists:
            self.assertFalse(
                any("barrier" in key.lower() for key in allowlist), allowlist
            )
        source = Path(analysis.__file__).read_text(encoding="utf-8")
        self.assertNotIn('saved["barrier', source)
        self.assertNotIn("saved['barrier", source)

    def test_complete_formal_chain_and_locked_no_dominator_rule(self):
        payload = analysis.analyze(self.report)
        self.assertEqual(
            payload["merged_alpha_grid"], [5.0, 7.5, 10.0, 12.5, 15.0, 20.0]
        )
        self.assertEqual(payload["joint_full_curve_dominator_alphas"], [])
        self.assertEqual(payload["alpha_lock_status"], "not_locked_no_joint_dominator")
        self.assertEqual(payload["stop_ablation_center_alpha"], 12.5)
        self.assertTrue(payload["stop_ablation_center_is_not_a_winner"])
        self.assertFalse(payload["winner_claim_made"])
        self.assertTrue(payload["barrier_remains_embargoed"])
        self.assertFalse(payload["barrier_data_accessed"])
        self.assertFalse(payload["barrier_computation_or_densification_performed"])
        self.assertEqual(
            payload["verified_input_report_member_counts"],
            {"refinement": 8, "coarse": 20, "merged_for_analysis": 12},
        )
        ranking = payload["official_tail_deficit_ranking"]
        self.assertEqual([row["alpha"] for row in ranking], [12.5, 7.5, 5.0, 10.0, 15.0, 20.0])
        self.assertEqual(ranking[0]["unreachable_official_tail_threshold_count"], 0)


if __name__ == "__main__":
    unittest.main()
