#!/usr/bin/env python3
"""Zero-OpenMM tests for promotion-bound execution authorization."""

import unittest

from issue_execution_authorization import (
    ALPHA_SCOPE_R2_SHA256,
    METRIC_POLICY_AMENDMENT_SHA256,
    validate_alpha_promotion,
    validate_infrastructure_retry,
    validate_r_controlled_scan,
)


def receipt(values=(0.1, 1.0, 10.0)):
    return {
        "status": "PASS_ALPHA_SELECTION_FROM_TWO_PRINCIPAL_DIAGNOSTICS",
        "metric_policy_amendment_sha256": METRIC_POLICY_AMENDMENT_SHA256,
        "alpha_scope_R2_sha256": ALPHA_SCOPE_R2_SHA256,
        "promoted_alpha_values_for_independent_990_N_F_runs": list(values),
    }


def registration(alpha=1.0, budget=990, tier="screen"):
    return {
        "alpha": alpha,
        "internal_nf_budget": budget,
        "tier": tier,
    }


class AlphaPromotionAuthorizationTests(unittest.TestCase):
    def test_accepts_exact_promoted_990_member(self):
        self.assertEqual(
            validate_alpha_promotion(receipt(), registration()),
            (0.1, 1.0, 10.0),
        )

    def test_rejects_unpromoted_alpha(self):
        with self.assertRaisesRegex(RuntimeError, "not promoted"):
            validate_alpha_promotion(receipt(), registration(alpha=100.0))

    def test_rejects_wrong_budget_or_tier(self):
        for member in (registration(budget=480), registration(tier="production")):
            with self.subTest(member=member):
                with self.assertRaisesRegex(RuntimeError, "990-N_F"):
                    validate_alpha_promotion(receipt(), member)

    def test_rejects_policy_hash_drift_and_duplicate_values(self):
        bad_hash = receipt()
        bad_hash["metric_policy_amendment_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "metric-policy"):
            validate_alpha_promotion(bad_hash, registration())
        with self.assertRaisesRegex(RuntimeError, "three distinct"):
            validate_alpha_promotion(
                receipt(values=(1.0, 1.0, 10.0)), registration()
            )

    def test_accepts_strict_infrastructure_replacement(self):
        failure = {
            "status": "RECORDED_INFRASTRUCTURE_FAILURE_NO_SCIENTIFIC_RESULT",
            "original_label": "old_r1",
            "replacement_label": "new_r2",
            "returncode": 120,
            "scientific_result_exists": False,
            "registered_controls": {
                "alpha": 1.0,
                "r": 0.5,
                "m_max": 5,
                "optimization_Number_of_Forces": 990,
            },
        }
        registered = {
            **registration(),
            "residual_ratio_r": 0.5,
            "inner_max_steps_m_max": 5,
        }
        validate_infrastructure_retry(failure, registered, "new_r2")
        failure["scientific_result_exists"] = True
        with self.assertRaisesRegex(RuntimeError, "absence of result"):
            validate_infrastructure_retry(failure, registered, "new_r2")

    def test_accepts_only_registered_r_controlled_member(self):
        selection = {
            "status": "PASS_ALPHA_CONFIRMATION_TWO_METRIC_SELECTION",
            "alpha_star_for_r_controlled_scan": 10.0,
        }
        efficiency = {
            "status": "OPERATIVE_PROSPECTIVE_EFFICIENCY_AMENDMENT",
            "r_controlled_scan": {
                "fixed_alpha": 10.0,
                "fixed_m_max": 5,
                "registered_budget_upper_bound": 480,
                "new_members": [{"r": value} for value in (0.1, 0.3, 0.7, 0.8)],
            },
        }
        member = {
            "tier": "screen",
            "internal_nf_budget": 480,
            "alpha": 10.0,
            "residual_ratio_r": 0.3,
            "inner_max_steps_m_max": 5,
        }
        self.assertEqual(
            validate_r_controlled_scan(selection, efficiency, member),
            (0.1, 0.3, 0.7, 0.8),
        )
        member["residual_ratio_r"] = 0.5
        with self.assertRaisesRegex(RuntimeError, "registered new r"):
            validate_r_controlled_scan(selection, efficiency, member)


if __name__ == "__main__":
    unittest.main()
