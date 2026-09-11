from __future__ import annotations

import unittest

import numpy as np

import analyze_stage4_ala_discrete_single_iperp_ctp as analysis


def point(nf: int, i_perp: float) -> dict[str, object]:
    return {"NF": nf, "i_perp": i_perp}


class IperpCostToPrecisionAnalysisTests(unittest.TestCase):
    def test_eligible_is_initial_plus_committed_then_strict_mesh_gate(self):
        points = analysis.eligible_single_points(
            cost=np.asarray([10, 20, 30, 40, 50]),
            i_perp=np.asarray([1.0, 0.8, 0.6, 0.4, 0.2]),
            committed=np.asarray([False, False, True, True, True]),
            cv=np.asarray([0.001, 0.001, 0.0065, 0.00651, 0.001]),
            ratio=np.asarray([1.001, 1.001, 1.03, 1.02, 1.031]),
        )
        self.assertEqual([row["NF"] for row in points], [10, 30])
        self.assertTrue(points[0]["is_initial"])
        self.assertFalse(points[0]["reparam_committed"])
        self.assertTrue(points[1]["reparam_committed"])

    def test_pareto_records_and_ctp_use_actual_cost_without_interpolation(self):
        points = [
            point(10, 1.0),
            point(20, 0.7),
            point(30, 0.8),
            point(40, 0.5),
        ]
        self.assertEqual(
            [(row["NF"], row["i_perp"]) for row in analysis.pareto_records(points)],
            [(10, 1.0), (20, 0.7), (40, 0.5)],
        )
        self.assertEqual(analysis.cost_to_precision(points, 0.75), 20)
        self.assertEqual(analysis.cost_to_precision(points, 0.6), 40)
        self.assertIsNone(analysis.cost_to_precision(points, 0.49))

    def test_full_dominance_requires_every_threshold_and_tightest(self):
        comparator = [point(10, 1.0), point(30, 0.5), point(50, 0.2)]
        candidate = [point(10, 0.9), point(20, 0.5), point(40, 0.2)]
        result = analysis.compare_curves(candidate, comparator, "control")
        self.assertTrue(
            result["weakly_no_slower_at_every_comparator_pareto_threshold"]
        )
        self.assertTrue(result["strictly_faster_at_least_one_threshold"])
        self.assertTrue(result["candidate_does_not_lose_tightest_precision"])
        self.assertTrue(result["full_curve_dominates"])

        no_strict = analysis.compare_curves(comparator, comparator, "control")
        self.assertFalse(no_strict["full_curve_dominates"])

    def test_unreachable_tight_threshold_is_an_exact_loss(self):
        comparator = [point(10, 1.0), point(30, 0.1)]
        candidate = [point(10, 0.9), point(20, 0.2)]
        result = analysis.compare_curves(candidate, comparator, "control")
        self.assertFalse(result["full_curve_dominates"])
        self.assertEqual(
            result["near_miss"]["candidate_unreachable_loss_count"], 1
        )
        self.assertEqual(result["near_miss"]["loss_threshold_count"], 1)
        self.assertGreater(result["near_miss"]["tightest_i_perp_excess"], 0.0)

    def test_official_anchor_scope_marks_high_i_without_comparing(self):
        candidate = [point(10, 0.1), point(20, 0.005)]
        official = [point(100, 0.006), point(200, 0.004)]
        rows = analysis.anchor_rows(
            candidate, official, [0.1, 0.005], domain_max=0.007
        )
        self.assertEqual(
            rows[0]["domain_status"],
            "outside_comparator_post_activation_domain",
        )
        self.assertEqual(rows[0]["outcome"], "not_compared")
        self.assertEqual(rows[1]["domain_status"], "inside_comparator_domain")
        self.assertEqual(rows[1]["outcome"], "strict_win")

    def test_access_allowlists_exclude_barrier_fields(self):
        allowlists = (
            analysis.SINGLE_ALLOWED_KEYS,
            analysis.OFFICIAL_ARCHIVE_ALLOWED_KEYS,
            analysis.OFFICIAL_COMBINED_ALLOWED_KEYS,
            analysis.OFFICIAL_COMPONENT_ALLOWED_KEYS,
        )
        for allowlist in allowlists:
            self.assertFalse(
                any("barrier" in key.lower() for key in allowlist), allowlist
            )


if __name__ == "__main__":
    unittest.main()
