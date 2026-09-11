from __future__ import annotations

import json
from pathlib import Path
import unittest

import analyze_stage4_ala_discrete_single_selected_eta_iperp_ctp as analysis


def point(nf: int, i_perp: float) -> dict[str, object]:
    return {"NF": nf, "i_perp": i_perp}


def deficit(values):
    return {
        "candidate_minus_comparator_NF_tight_to_loose": list(values),
        "candidate_unreachable_tight_to_loose": [value is None for value in values],
    }


class SelectedEtaIperpCtpAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.report = cls.root / "reports" / (
            "stage4_ala_discrete_single_selected_eta_pilot_v1_"
            "N254_r1_1787256460847550000.json"
        )
        cls.prereg = cls.root / "reports" / (
            "stage4_ala_discrete_single_selected_eta_pilot_v1_"
            "N254_r1_1787256460847550000_preregistered.json"
        )

    def test_npz_allowlists_and_source_exclude_barrier_access(self):
        for name, keys in analysis.NPZ_ACCESS_ALLOWLISTS.items():
            self.assertFalse(any("barrier" in key.lower() for key in keys), name)
        source = Path(analysis.__file__).read_text(encoding="utf-8")
        self.assertNotIn('saved["barrier', source)
        self.assertNotIn("saved['barrier", source)

    def test_actual_nf_without_interpolation(self):
        points = [point(10, 1.0), point(40, 0.2), point(90, 0.05)]
        self.assertEqual(analysis.base.cost_to_precision(points, 0.2), 40)
        self.assertEqual(analysis.base.cost_to_precision(points, 0.1), 90)
        self.assertIsNone(analysis.base.cost_to_precision(points, 0.01))

    def test_eligibility_is_initial_or_committed_then_mesh(self):
        curve = analysis.build_eligible_points(
            cost=[10, 20, 30, 40, 50],
            i_perp=[1.0, 0.8, 0.6, 0.4, 0.2],
            committed=[False, False, True, True, True],
            cv=[0.0065, 0.0, 0.0065, 0.0065001, 0.0],
            mesh_ratio=[1.03, 1.0, 1.03, 1.0, 1.0300001],
        )
        self.assertEqual(curve["complete_state_NF"], [10, 30, 40, 50])
        self.assertEqual(curve["mesh_eligible_complete_state_NF"], [10, 30])

    def test_wtlu_keeps_unreachable_separate_and_as_loss(self):
        comparator = [
            point(10, 1.0),
            point(20, 0.5),
            point(30, 0.2),
            point(40, 0.1),
        ]
        candidate = [point(5, 0.8), point(20, 0.5), point(35, 0.2)]
        result = analysis.comparison_with_wtlu(
            candidate, comparator, "synthetic"
        )
        self.assertEqual(result["W_T_L_U"], {"W": 1, "T": 1, "L": 1, "U": 1})
        self.assertEqual(result["loss_count_including_unreachable"], 2)
        self.assertFalse(result["full_curve_dominates"])

    def test_primary_requires_all_three_comparators(self):
        comparisons = [
            {
                "eta": 1.25,
                "joint_full_curve_dominates_all_three": False,
                "eta_one_deficit_vector": deficit((-2,)),
                "official_deficit_vector": deficit((-3,)),
                "legacy_deficit_vector": deficit((-4,)),
            }
        ]
        decision = analysis.decide_primary_winner(comparisons)
        self.assertEqual(decision["joint_full_curve_dominator_etas"], [])
        self.assertIsNone(decision["selected_primary_winner_eta"])
        self.assertFalse(decision["winner_claim_made"])
        self.assertFalse(decision["eta_lock_allowed"])
        self.assertFalse(decision["barrier_unembargo_allowed"])

    def test_registered_tiebreak_is_eta1_then_official_legacy_eta(self):
        def row(eta, eta1, official, legacy):
            return {
                "eta": eta,
                "joint_full_curve_dominates_all_three": True,
                "eta_one_deficit_vector": deficit(eta1),
                "official_deficit_vector": deficit(official),
                "legacy_deficit_vector": deficit(legacy),
            }

        decision = analysis.decide_primary_winner(
            [
                row(1.25, (-5, -1), (100, 100), (100, 100)),
                row(1.5, (-4, -99), (-100, -100), (-100, -100)),
            ]
        )
        self.assertEqual(decision["selected_primary_winner_eta"], 1.25)

        exact = analysis.decide_primary_winner(
            [
                row(1.5, (-1,), (-1,), (-1,)),
                row(1.25, (-1,), (-1,), (-1,)),
            ]
        )
        self.assertEqual(exact["selected_primary_winner_eta"], 1.25)

    def test_unreachable_deficit_is_null_and_sorted_tight_to_loose(self):
        vector = analysis.deficit_vector(
            [point(20, 0.2)],
            [point(10, 0.5), point(30, 0.1)],
        )
        self.assertEqual(vector["thresholds_tight_to_loose"], [0.1, 0.5])
        self.assertEqual(
            vector["candidate_minus_comparator_NF_tight_to_loose"],
            [None, 10],
        )
        self.assertEqual(
            vector["candidate_unreachable_tight_to_loose"], [True, False]
        )
        json.dumps(vector, allow_nan=False)

    def test_preregistration_structure_and_hash_manifests(self):
        prereg = analysis.read_json(self.prereg)
        analysis.validate_preregistration(prereg, self.report)
        analysis.verify_hash_manifest(prereg["run_source_manifest"], "sources")
        analysis.verify_hash_manifest(prereg["decision_inputs"], "inputs")

    def test_complete_formal_chain_is_strict_json_and_consistent(self):
        payload = analysis.analyze(self.report)
        json.dumps(payload, allow_nan=False)
        self.assertEqual(payload["eta_grid"], [1.0, 1.25, 1.5, 2.0])
        self.assertEqual(payload["candidate_etas"], [1.25, 1.5, 2.0])
        self.assertTrue(payload["new_eta_one_ctp_parity_with_oracle"])
        self.assertEqual(len(payload["comparisons"]), 3)
        self.assertFalse(payload["barrier_data_accessed"])
        self.assertFalse(payload["barrier_computation_or_densification_performed"])
        self.assertFalse(payload["barrier_unembargoed"])
        for row in payload["comparisons"]:
            for key in (
                "same_build_eta1_comparison",
                "same_alpha_legacy_comparison",
                "official_stage4_v4_comparison",
            ):
                comparison = row[key]
                counts = comparison["W_T_L_U"]
                self.assertEqual(
                    sum(counts.values()),
                    comparison["comparator_pareto_record_count"],
                )
        if payload["selected_primary_winner_eta"] is None:
            self.assertFalse(payload["winner_claim_made"])
            self.assertFalse(payload["eta_lock_allowed"])
            self.assertFalse(payload["barrier_unembargo_allowed"])
            self.assertTrue(payload["barrier_remains_embargoed"])


if __name__ == "__main__":
    unittest.main()
