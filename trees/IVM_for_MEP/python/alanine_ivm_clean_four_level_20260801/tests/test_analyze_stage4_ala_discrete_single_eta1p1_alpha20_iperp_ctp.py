from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from unittest import mock
import unittest

import numpy as np

import analyze_stage4_ala_discrete_single_eta1p1_alpha20_iperp_ctp as analysis


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


class Eta1p1Alpha20IperpCtpAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.report = cls.root / "reports" / (
            "stage4_ala_discrete_single_eta1p1_alpha20_followup_v1_"
            "N254_r1_1787258614120573000.json"
        )
        cls.preregistration = cls.root / "reports" / (
            "stage4_ala_discrete_single_eta1p1_alpha20_followup_v1_"
            "N254_r1_1787258614120573000_preregistered.json"
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

    def test_preregistration_matches_frozen_alpha20_and_secondary_schema(self):
        prereg = analysis.read_json(self.preregistration)
        analysis.validate_preregistration(prereg, self.report)
        self.assertEqual(
            set(prereg["decision_inputs"]), analysis.EXPECTED_DECISION_INPUT_KEYS
        )
        self.assertEqual(
            set(prereg["run_source_manifest"]), analysis.EXPECTED_RUN_SOURCE_KEYS
        )
        self.assertEqual(prereg["fixed_alpha"], 20.0)
        self.assertEqual(prereg["fixed_eta"], 1.1)
        self.assertFalse(prereg["same_eta_alpha15_control_in_primary_success_gate"])
        self.assertTrue(prereg["same_eta_alpha15_alpha_only_difference_verified"])

        drifted = copy.deepcopy(prereg)
        drifted["same_eta_alpha15_control_in_primary_success_gate"] = True
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

    def test_wtlu_counts_unreachable_as_formal_loss(self):
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

    def test_primary_gate_is_exact_triple_and_excludes_secondary_alpha15(self):
        names = {
            "same_alpha_eta_one_candidate",
            "same_alpha_legacy",
            "official_stage4_v4",
        }
        comparisons = {name: {"full_curve_dominates": True} for name in names}
        self.assertTrue(analysis.primary_success(comparisons))
        comparisons["official_stage4_v4"]["full_curve_dominates"] = False
        self.assertFalse(analysis.primary_success(comparisons))
        with self.assertRaises(RuntimeError):
            analysis.primary_success(
                {
                    **comparisons,
                    "same_eta_alpha15": {"full_curve_dominates": True},
                }
            )

    def test_formal_analysis_tracks_every_npz_key_and_never_reads_barrier(self):
        accessed: list[str] = []
        local_json_reads: list[Path] = []
        real_load = np.load
        real_read_json = analysis.read_json

        def tracking_load(*args, **kwargs):
            loaded = real_load(*args, **kwargs)
            if hasattr(loaded, "files"):
                return _TrackedNpz(loaded, accessed)
            return loaded

        def tracking_read_json(path):
            local_json_reads.append(Path(path).resolve())
            return real_read_json(path)

        with mock.patch.object(np, "load", side_effect=tracking_load), mock.patch.object(
            analysis, "read_json", side_effect=tracking_read_json
        ):
            payload = analysis.analyze(self.report)

        allowed = set().union(*analysis.NPZ_ACCESS_ALLOWLISTS.values())
        self.assertTrue(accessed)
        self.assertTrue(set(accessed).issubset(allowed), sorted(set(accessed) - allowed))
        self.assertFalse(any("barrier" in key.lower() for key in accessed))
        self.assertFalse(any("densif" in key.lower() for key in accessed))
        json.dumps(payload, allow_nan=False)

        prereg = analysis.read_json(self.preregistration)
        coarse_audits = {
            Path(prereg["decision_inputs"][name]["path"]).resolve()
            for name in (
                "same_alpha_eta_one_candidate_audit",
                "same_alpha_legacy_audit",
            )
        }
        self.assertTrue(coarse_audits.isdisjoint(local_json_reads))
        self.assertTrue(payload["coarse_comparator_audits_hash_verified_but_not_decoded"])

        self.assertEqual(payload["fixed_alpha"], 20.0)
        self.assertEqual(payload["fixed_eta"], 1.1)
        self.assertEqual(
            set(payload["comparisons"]),
            {
                "same_alpha_eta_one_candidate",
                "same_alpha_legacy",
                "official_stage4_v4",
            },
        )
        for comparison in payload["comparisons"].values():
            self.assertEqual(
                sum(comparison["W_T_L_U"].values()),
                comparison["comparator_pareto_record_count"],
            )
        expected_wtlu = {
            "same_alpha_eta_one_candidate": {"W": 2, "T": 2, "L": 4, "U": 0},
            "same_alpha_legacy": {"W": 1, "T": 2, "L": 6, "U": 3},
            "official_stage4_v4": {"W": 0, "T": 0, "L": 7, "U": 1},
        }
        self.assertEqual(
            {
                name: row["W_T_L_U"]
                for name, row in payload["comparisons"].items()
            },
            expected_wtlu,
        )
        self.assertEqual(
            payload["candidate"]["pareto_records"][-1]["i_perp"],
            0.00021906195086532175,
        )
        secondary = payload["secondary_same_eta_alpha15"]
        self.assertFalse(secondary["in_primary_success_gate"])
        self.assertTrue(secondary["alpha_only_command_difference"]["verified"])
        self.assertEqual(secondary["comparator"]["alpha"], 15.0)
        self.assertEqual(
            secondary["comparison"]["W_T_L_U"],
            {"W": 0, "T": 2, "L": 12, "U": 5},
        )
        self.assertFalse(payload["primary_success_criterion_satisfied"])
        self.assertFalse(payload["winner_claim_made"])
        self.assertFalse(payload["barrier_data_accessed"])
        self.assertFalse(payload["barrier_computation_or_densification_performed"])
        self.assertFalse(payload["barrier_unembargoed"])
        self.assertTrue(payload["barrier_remains_embargoed"])
        self.assertFalse(payload["barrier_unembargo_allowed_by_this_analysis"])

    def test_write_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            payload = {
                "barrier_data_accessed": False,
                "barrier_computation_or_densification_performed": False,
                "barrier_unembargoed": False,
                "barrier_remains_embargoed": True,
            }
            analysis.write_exclusive(payload, output)
            with self.assertRaises(FileExistsError):
                analysis.write_exclusive(payload, output)


if __name__ == "__main__":
    unittest.main()
