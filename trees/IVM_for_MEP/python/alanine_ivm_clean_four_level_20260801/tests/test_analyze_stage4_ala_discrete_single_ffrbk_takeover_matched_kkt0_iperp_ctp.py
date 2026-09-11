from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import analyze_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_iperp_ctp as analysis
import analyze_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_iperp_ctp as matched


def point(nf: int, value: float) -> dict[str, object]:
    return {"NF": nf, "i_perp": value}


def synthetic_payload(arm: str, n: int = 4) -> dict[str, object]:
    implementation = f"synthetic+{arm}"
    payload: dict[str, object] = {
        "active_alpha": np.full(n, 15.0),
        "alpha": np.array(15.0),
        "alpha_mode": np.array("fixed"),
        "configuration_sha256": np.array("synthetic-configuration"),
        "discrete_all_returned_states_evaluated": np.array(True),
        "discrete_history": np.array(7),
        "discrete_implementation_id": np.array(implementation),
        "discrete_inner_max_steps": np.array(10),
        "discrete_residual_ratio": np.array(0.8),
        "discrete_stop_mode": np.array("r08c10"),
        "frozen_subproblem": np.array("discrete_b040"),
        "i_perp": np.linspace(1.0, 0.1, n),
        "initial_path_sha256": np.array(analysis.COMMON_INITIAL_ARRAY_SHA256),
        "plbfgs_kkt_atol": np.array(0.0),
        "reparam_committed": np.array([False] + [True] * (n - 1)),
        "reparam_mode": np.array("state_curve"),
        "segment_length_cv": np.zeros(n),
        "segment_length_ratio": np.ones(n),
        "stop_complete_iperp": np.array(-1.0),
        "stop_reason": np.array("cached_coordinate_stall"),
        "total_internal_joint_evaluations": np.arange(1, n + 1) * 20,
        "discrete_ffrbk_selector_aborted_selection_count": np.array(0),
        "discrete_ffrbk_selector_aborted_solver_calls": np.array(0),
        "discrete_ffrbk_selector_alpha": np.array(15.0),
        "discrete_ffrbk_selector_eta": np.array(1.1),
        "discrete_ffrbk_selector_extra_physical_evaluations": np.array(0),
        "discrete_ffrbk_selector_final_implementation_id": np.array(implementation),
        "discrete_ffrbk_selector_h0_shadow_constructed": np.array(True),
        "discrete_ffrbk_selector_history": np.array(7),
        "discrete_ffrbk_selector_inner_max_steps": np.array(10),
        "discrete_ffrbk_selector_invalid_lbfgs_h0_fallback_enabled": np.array(True),
        "discrete_ffrbk_selector_kkt_atol": np.array(0.0),
        "discrete_ffrbk_selector_mode": np.array(arm),
        "discrete_ffrbk_selector_mode_policy_id": np.array(
            analysis.MODE_POLICY_IDS[arm]
        ),
        "discrete_ffrbk_selector_one_way_latch_is_preference": np.array(
            arm == "one_way_takeover"
        ),
        "discrete_ffrbk_selector_operator_policy_id": np.array(
            "frozen-ffrbk-h0-and-operator-initialized-lbfgs-v1"
        ),
        "discrete_ffrbk_selector_operator_preserved": np.array(True),
        "discrete_ffrbk_selector_post_select_eta_composed": np.array(True),
        "discrete_ffrbk_selector_reparameterization_inside_subproblem": np.array(False),
        "discrete_ffrbk_selector_residual_ratio": np.array(0.8),
        "discrete_ffrbk_selector_runtime_policy_id": np.array(
            "ala-discrete-ffrbk-selector-runtime-v1"
        ),
        "discrete_ffrbk_selector_selector_only_change": np.array(True),
        "discrete_ffrbk_selector_stop_mode": np.array("r08c10"),
        "discrete_ffrbk_selector_true_lbfgs_only_construction": np.array(False),
        "discrete_ffrbk_selector_uncommitted_selection_count": np.array(0),
    }
    return payload


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


class TakeoverIperpCtpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.report = cls.root / "reports" / (
            "stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_"
            "N254_r1_1787262084320785000.json"
        )

    def test_complete_then_mesh_gate_is_inclusive(self):
        curve = matched.build_observed_eligible_curve(
            cost=[10, 20, 30, 40, 50],
            i_perp=[1.0, 0.8, 0.6, 0.4, 0.2],
            committed=[False, False, True, True, True],
            cv=[0.0065, 0.0, 0.0065, 0.0065001, 0.0],
            mesh_ratio=[1.03, 1.0, 1.03, 1.0, 1.0300001],
        )
        self.assertEqual(curve["complete_state_NF"], [10, 30, 40, 50])
        self.assertEqual(curve["mesh_eligible_complete_state_NF"], [10, 30])

    def test_unreachable_is_U_formal_loss_without_interpolation(self):
        candidate = [point(20, 1.0), point(100, 0.2)]
        comparator = [
            point(20, 1.0),
            point(80, 0.5),
            point(160, 0.2),
            point(300, 0.1),
        ]
        row = matched.comparison_with_wtlu(candidate, comparator, "comparator")
        self.assertEqual(row["W_T_L_U"], {"W": 1, "T": 1, "L": 1, "U": 1})
        self.assertFalse(row["full_curve_dominates"])

    def test_identical_pair_is_bidirectional_ties_not_a_dominance(self):
        curve = [point(20, 1.0), point(100, 0.2), point(200, 0.1)]
        pair = analysis.compare_bidirectional(curve, list(curve))
        self.assertTrue(pair["eligible_curves_exactly_identical"])
        self.assertEqual(
            pair["one_way_takeover_vs_forced_operator_lbfgs_selection"]["W_T_L_U"],
            {"W": 0, "T": 3, "L": 0, "U": 0},
        )
        self.assertEqual(pair["directional_full_curve_dominators"], [])

    def test_winner_requires_all_three_full_curves_and_tightest(self):
        strong = [point(10, 1.0), point(90, 0.1)]
        weak = [point(30, 1.0), point(120, 0.2)]
        candidates = {
            "one_way_takeover": strong,
            "forced_operator_lbfgs_selection": [point(10, 1.0)],
        }
        performance = analysis.performance_comparisons(
            candidates, weak, weak, weak
        )
        self.assertTrue(performance["one_way_takeover"]["qualifies"])
        self.assertFalse(
            performance["forced_operator_lbfgs_selection"]["qualifies"]
        )
        pair = analysis.compare_bidirectional(
            candidates["one_way_takeover"],
            candidates["forced_operator_lbfgs_selection"],
        )
        winner = analysis.decide_winner(performance, pair)
        self.assertEqual(winner["locked_winner"], "one_way_takeover")

    def test_variable_length_loader_uses_only_registered_fields(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for arm in analysis.ARMS:
                path = root / f"{arm}.npz"
                np.savez(path, **synthetic_payload(arm))
                loaded = analysis.load_arm_result(
                    path,
                    arm,
                    expected_sha256=analysis.file_sha256(path),
                    expected_record_count=4,
                )
                self.assertEqual(loaded["observed_trajectory_length"], 4)
                self.assertTrue(loaded["terminated_before_N254"])
                self.assertFalse(loaded["unobserved_tail_extrapolated"])

    def test_npz_allowlist_and_direct_indexing_exclude_physical_arrays(self):
        forbidden = ("barrier", "densif", "potential_energy", "energies", "stored_path")
        self.assertFalse(
            any(
                any(token in key.lower() for token in forbidden)
                for key in analysis.NPZ_ACCESS_ALLOWLIST
            )
        )
        source = Path(analysis.__file__).read_text(encoding="utf-8")
        for quote in ('saved["', "saved['"):
            for token in forbidden:
                self.assertNotIn(f"{quote}{token}", source)

    def test_real_report_provenance_and_formal_result(self):
        accessed: list[str] = []
        real_load = np.load

        def tracking_load(*args, **kwargs):
            loaded = real_load(*args, **kwargs)
            if hasattr(loaded, "files"):
                return _TrackedNpz(loaded, accessed)
            return loaded

        with mock.patch.object(np, "load", side_effect=tracking_load):
            payload = analysis.analyze(self.report)

        self.assertTrue(accessed)
        self.assertTrue(set(accessed).issubset(analysis.NPZ_ACCESS_ALLOWLIST))
        json.dumps(payload, allow_nan=False)
        self.assertEqual(
            {
                arm: row["observed_trajectory_length"]
                for arm, row in payload["arms"].items()
            },
            {"one_way_takeover": 254, "forced_operator_lbfgs_selection": 254},
        )
        pair = payload["arm_bidirectional_comparison"]
        self.assertTrue(pair["eligible_curves_exactly_identical"])
        self.assertTrue(pair["ctp_trajectory_sha256_equal"])
        self.assertEqual(
            pair["one_way_takeover_vs_forced_operator_lbfgs_selection"]["W_T_L_U"],
            {"W": 0, "T": 18, "L": 0, "U": 0},
        )
        for arm in analysis.ARMS:
            rows = payload["performance_candidate_comparisons"][arm]["comparisons"]
            self.assertEqual(
                rows["matched_current0"]["W_T_L_U"],
                {"W": 10, "T": 4, "L": 5, "U": 0},
            )
            self.assertEqual(
                rows["same_alpha_legacy"]["W_T_L_U"],
                {"W": 2, "T": 3, "L": 6, "U": 2},
            )
            self.assertEqual(
                rows["official_stage4_v4"]["W_T_L_U"],
                {"W": 4, "T": 0, "L": 4, "U": 0},
            )
            self.assertFalse(payload["performance_candidate_comparisons"][arm]["qualifies"])
        self.assertEqual(payload["performance_qualifiers"], [])
        self.assertFalse(payload["winner_claim_made"])
        self.assertFalse(payload["physical_objective_arrays_accessed"])
        self.assertFalse(payload["barrier_arrays_accessed"])
        self.assertFalse(payload["densification_performed"])


if __name__ == "__main__":
    unittest.main()
