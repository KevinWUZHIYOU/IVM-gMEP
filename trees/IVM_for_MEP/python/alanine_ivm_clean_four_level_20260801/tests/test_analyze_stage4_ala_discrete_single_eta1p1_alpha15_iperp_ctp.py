from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest import mock
import unittest

import numpy as np

import analyze_stage4_ala_discrete_single_eta1p1_alpha15_iperp_ctp as analysis


def point(nf: int, i_perp: float) -> dict[str, object]:
    return {"NF": nf, "i_perp": i_perp}


class _TrackedNpz:
    def __init__(self, saved, accessed: list[str]):
        self._saved = saved
        self._accessed = accessed

    @property
    def files(self):
        return self._saved.files

    def __getitem__(self, key):
        self._accessed.append(str(key))
        return self._saved[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._saved.close()
        return False


class Eta1p1Alpha15IperpCtpAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.report = cls.root / "reports" / (
            "stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_"
            "N254_r1_1787257808883811000.json"
        )
        cls.preregistration = cls.root / "reports" / (
            "stage4_ala_discrete_single_eta1p1_alpha15_followup_v1_"
            "N254_r1_1787257808883811000_preregistered.json"
        )

    def test_npz_allowlists_exclude_barrier_and_densified_fields(self):
        for name, keys in analysis.NPZ_ACCESS_ALLOWLISTS.items():
            self.assertFalse(
                any(
                    "barrier" in key.lower() or "densif" in key.lower()
                    for key in keys
                ),
                name,
            )
        source = Path(analysis.__file__).read_text(encoding="utf-8")
        self.assertNotIn('saved["barrier', source)
        self.assertNotIn("saved['barrier", source)
        self.assertNotIn('saved["densif', source)
        self.assertNotIn("saved['densif", source)

    def test_schema_aware_barrier_metadata_accepts_only_frozen_embargo(self):
        accepted = {
            "barrier_fields_accessed_by_wrapper_or_launcher": False,
            "sparse_barrier_role": "embargoed_not_read_not_reported_not_ranked",
            "densified_energy_barrier_role": "embargoed_for_entire_followup",
        }
        analysis.assert_false_barrier_metadata(accepted)
        for key, value in (
            ("barrier_fields_accessed_by_wrapper_or_launcher", True),
            ("sparse_barrier_role", "used_for_ranking"),
            ("unknown_barrier_mode", False),
        ):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                analysis.assert_false_barrier_metadata({key: value})

    def test_preregistration_matches_the_frozen_single_arm_schema(self):
        prereg = analysis.read_json(self.preregistration)
        analysis.validate_preregistration(prereg, self.report)
        self.assertEqual(set(prereg["decision_inputs"]), analysis.EXPECTED_DECISION_INPUT_KEYS)
        self.assertEqual(set(prereg["run_source_manifest"]), analysis.EXPECTED_RUN_SOURCE_KEYS)
        self.assertEqual(prereg["fixed_alpha"], 15.0)
        self.assertEqual(prereg["fixed_eta"], 1.1)

        drifted = copy.deepcopy(prereg)
        drifted["primary_comparators"][0] = "alpha12.5"
        with self.assertRaises(RuntimeError):
            analysis.validate_preregistration(drifted, self.report)

    def test_eligibility_is_initial_or_committed_then_mesh(self):
        curve = analysis.eta_base.build_eligible_points(
            cost=[10, 20, 30, 40, 50],
            i_perp=[1.0, 0.8, 0.6, 0.4, 0.2],
            committed=[False, False, True, True, True],
            cv=[0.0065, 0.0, 0.0065, 0.0065001, 0.0],
            mesh_ratio=[1.03, 1.0, 1.03, 1.0, 1.0300001],
        )
        self.assertEqual(curve["complete_state_NF"], [10, 30, 40, 50])
        self.assertEqual(curve["mesh_eligible_complete_state_NF"], [10, 30])

    def test_ctp_uses_minimum_observed_nf_without_interpolation(self):
        points = [point(10, 1.0), point(40, 0.2), point(90, 0.05)]
        self.assertEqual(analysis.base.cost_to_precision(points, 0.2), 40)
        self.assertEqual(analysis.base.cost_to_precision(points, 0.1), 90)
        self.assertIsNone(analysis.base.cost_to_precision(points, 0.01))

    def test_wtlu_counts_unreachable_as_a_formal_loss(self):
        comparator = [
            point(10, 1.0),
            point(20, 0.5),
            point(30, 0.2),
            point(40, 0.1),
        ]
        candidate = [point(5, 0.8), point(20, 0.5), point(35, 0.2)]
        result = analysis.comparison_with_wtlu(candidate, comparator, "synthetic")
        self.assertEqual(result["W_T_L_U"], {"W": 1, "T": 1, "L": 1, "U": 1})
        self.assertEqual(result["loss_count_including_unreachable"], 2)
        self.assertFalse(result["full_curve_dominates"])

    def test_triple_success_requires_exactly_all_three_dominances(self):
        names = {
            "same_alpha_eta_one_candidate",
            "same_alpha_legacy",
            "official_stage4_v4",
        }
        comparisons = {name: {"full_curve_dominates": True} for name in names}
        self.assertTrue(analysis.triple_success(comparisons))
        comparisons["official_stage4_v4"]["full_curve_dominates"] = False
        self.assertFalse(analysis.triple_success(comparisons))
        with self.assertRaises(RuntimeError):
            analysis.triple_success({"same_alpha_legacy": {"full_curve_dominates": True}})

    def test_formal_analysis_tracks_every_npz_key_and_never_reads_barrier(self):
        accessed: list[str] = []
        real_load = np.load

        def tracking_load(*args, **kwargs):
            loaded = real_load(*args, **kwargs)
            if hasattr(loaded, "files"):
                return _TrackedNpz(loaded, accessed)
            return loaded

        with mock.patch.object(np, "load", side_effect=tracking_load):
            payload = analysis.analyze(self.report)

        allowed = set().union(*analysis.NPZ_ACCESS_ALLOWLISTS.values())
        self.assertTrue(accessed)
        self.assertTrue(set(accessed).issubset(allowed), sorted(set(accessed) - allowed))
        self.assertFalse(any("barrier" in key.lower() for key in accessed))
        self.assertFalse(any("densif" in key.lower() for key in accessed))
        json.dumps(payload, allow_nan=False)

        self.assertEqual(payload["fixed_alpha"], 15.0)
        self.assertEqual(payload["fixed_eta"], 1.1)
        self.assertEqual(set(payload["comparisons"]), {
            "same_alpha_eta_one_candidate",
            "same_alpha_legacy",
            "official_stage4_v4",
        })
        expected_wtlu = {
            "same_alpha_eta_one_candidate": {"W": 5, "T": 3, "L": 2, "U": 0},
            "same_alpha_legacy": {"W": 1, "T": 3, "L": 7, "U": 2},
            "official_stage4_v4": {"W": 1, "T": 0, "L": 7, "U": 0},
        }
        for comparison in payload["comparisons"].values():
            self.assertEqual(
                sum(comparison["W_T_L_U"].values()),
                comparison["comparator_pareto_record_count"],
            )
        self.assertEqual(
            {
                name: row["W_T_L_U"]
                for name, row in payload["comparisons"].items()
            },
            expected_wtlu,
        )
        self.assertEqual(
            payload["candidate"]["pareto_records"][-1]["i_perp"],
            0.00016186039155354832,
        )
        self.assertFalse(payload["primary_success_criterion_satisfied"])
        self.assertFalse(payload["winner_claim_made"])
        self.assertFalse(payload["barrier_data_accessed"])
        self.assertFalse(payload["barrier_computation_or_densification_performed"])
        self.assertFalse(payload["barrier_unembargoed"])
        self.assertTrue(payload["barrier_remains_embargoed"])
        self.assertFalse(payload["barrier_unembargo_allowed_by_this_analysis"])


if __name__ == "__main__":
    unittest.main()
