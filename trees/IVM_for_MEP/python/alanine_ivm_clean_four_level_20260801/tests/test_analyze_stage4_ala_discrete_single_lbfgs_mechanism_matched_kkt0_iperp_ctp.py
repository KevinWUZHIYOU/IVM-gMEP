from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import analyze_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_iperp_ctp as analysis


def point(nf: int, i_perp: float) -> dict[str, object]:
    return {"NF": nf, "i_perp": i_perp}


def synthetic_arm_payload(
    arm: str,
    *,
    n: int,
    stop_reason: str,
) -> dict[str, object]:
    cost = np.arange(1, n + 1, dtype=int) * 20
    payload: dict[str, object] = {
        "active_alpha": np.full(n, 15.0),
        "alpha": np.array(15.0),
        "alpha_mode": np.array("fixed"),
        "configuration_sha256": np.array("synthetic-configuration"),
        "discrete_all_returned_states_evaluated": np.array(True),
        "discrete_history": np.array(7),
        "discrete_implementation_id": np.array(
            "ala-frozen-discrete-ffrbk-dsel-m7-evaluated-v1"
        ),
        "discrete_inner_max_steps": np.array(10),
        "discrete_residual_ratio": np.array(0.8),
        "discrete_stop_mode": np.array("r08c10"),
        "frozen_subproblem": np.array("discrete_b040"),
        "i_perp": np.linspace(1.0, 0.1, n),
        "initial_path_sha256": np.array(
            analysis.COMMON_INITIAL_ARRAY_SHA256
        ),
        "plbfgs_kkt_atol": np.array(0.0),
        "reparam_committed": np.array([False] + [True] * (n - 1)),
        "reparam_mode": np.array("state_curve"),
        "segment_length_cv": np.zeros(n),
        "segment_length_ratio": np.ones(n),
        "state_curve_direct_objective_acceptance": np.array("per_image"),
        "stop_complete_iperp": np.array(-1.0),
        "stop_reason": np.array(stop_reason),
        "total_internal_joint_evaluations": cost,
    }
    if arm in analysis.PERFORMANCE_CANDIDATES:
        direct = arm == "direct_lbfgs_only"
        implementation = (
            "base+direct-lbfgs-only-no-selector-v1"
            if direct
            else "base+scalar-h0-vs-lbfgs-original-selector-v1"
        )
        payload.update(
            {
                "discrete_implementation_id": np.array(implementation),
                "discrete_scalar_lbfgs_direct_only": np.array(direct),
                "discrete_scalar_lbfgs_ffrbk_applied_inside_fixed_solver": np.array(False),
                "discrete_scalar_lbfgs_ffrbk_solve_normal_calls_inside_fixed_solver": np.array(0),
                "discrete_scalar_lbfgs_fixed_solver_preconditioner_is_none": np.array(True),
                "discrete_scalar_lbfgs_implementation_id": np.array(implementation),
                "discrete_scalar_lbfgs_mode": np.array(arm),
                "discrete_scalar_lbfgs_selector_competition": np.array(not direct),
            }
        )
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


class MatchedMechanismIperpCtpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.original_report = cls.root / "reports" / (
            "stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
            "N254_r1_1787260497802476000.json"
        )
        cls.direct_report = cls.root / "reports" / (
            "stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_"
            "direct_continuation_v1_N254_r1_1787261109219817000.json"
        )

    def test_variable_length_complete_then_mesh_gate(self):
        curve = analysis.build_observed_eligible_curve(
            cost=[10, 20, 30, 40, 50],
            i_perp=[1.0, 0.8, 0.6, 0.4, 0.2],
            committed=[False, False, True, True, True],
            cv=[0.0065, 0.0, 0.0065, 0.0065001, 0.0],
            mesh_ratio=[1.03, 1.0, 1.03, 1.0, 1.0300001],
        )
        self.assertEqual(curve["observed_trajectory_length"], 5)
        self.assertEqual(curve["complete_state_NF"], [10, 30, 40, 50])
        self.assertEqual(curve["mesh_eligible_complete_state_NF"], [10, 30])
        with self.assertRaises(RuntimeError):
            analysis.build_observed_eligible_curve(
                np.arange(255),
                np.ones(255),
                np.ones(255, dtype=bool),
                np.zeros(255),
                np.ones(255),
            )

    def test_early_stall_tail_is_U_without_extrapolation(self):
        scalar_prefix = [point(20, 1.0), point(100, 0.2)]
        current = [
            point(20, 1.0),
            point(80, 0.5),
            point(160, 0.2),
            point(300, 0.1),
            point(500, 0.01),
        ]
        compared = analysis.comparison_with_wtlu(
            scalar_prefix, current, "current"
        )
        self.assertEqual(compared["W_T_L_U"], {"W": 1, "T": 1, "L": 1, "U": 2})
        self.assertEqual(
            [
                row["outcome"]
                for row in compared["threshold_results"]
                if row["threshold_i_perp"] < 0.2
            ],
            ["candidate_unreachable_loss", "candidate_unreachable_loss"],
        )
        self.assertFalse(compared["unobserved_tail_extrapolated"])
        self.assertFalse(compared["full_curve_dominates"])

    def test_all_three_mechanism_pairs_are_bidirectional(self):
        curves = {
            "current_ffrbk_selector": [point(10, 1.0), point(100, 0.1)],
            "scalar_selector": [point(10, 1.0), point(80, 0.2)],
            "direct_lbfgs_only": [point(10, 1.0), point(60, 0.15)],
        }
        output = analysis.bidirectional_mechanism_comparisons(curves)
        self.assertEqual(
            set(output),
            {"ffrbk_operator_effect", "selector_effect", "combined_effect"},
        )
        for row in output.values():
            left = row["left_arm"]
            right = row["right_arm"]
            forward = row[f"{left}_vs_{right}"]
            reverse = row[f"{right}_vs_{left}"]
            self.assertIn("W_T_L_U", forward)
            self.assertIn("W_T_L_U", reverse)

    def test_A_and_B_each_require_exactly_three_full_dominances(self):
        curves = {
            "current_ffrbk_selector": [point(20, 1.0), point(100, 0.1)],
            "scalar_selector": [point(10, 1.0), point(90, 0.1)],
            "direct_lbfgs_only": [point(10, 1.0)],
        }
        legacy = [point(30, 1.0), point(110, 0.1)]
        official = [point(40, 1.0), point(120, 0.1)]
        output = analysis.performance_comparisons(curves, legacy, official)
        self.assertTrue(output["scalar_selector"]["full_curve_dominates_all_three"])
        self.assertFalse(output["direct_lbfgs_only"]["full_curve_dominates_all_three"])
        for arm in analysis.PERFORMANCE_CANDIDATES:
            self.assertEqual(
                set(output[arm]["comparisons"]), set(analysis.PRIMARY_COMPARATORS)
            )

    def test_variable_length_npz_loader_accepts_real_early_stop_semantics(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for arm in analysis.ARMS:
                n = 4 if arm != "current_ffrbk_selector" else 254
                reason = (
                    "evaluation_budget"
                    if n == 254
                    else "cached_coordinate_stall"
                )
                path = root / f"{arm}.npz"
                np.savez(path, **synthetic_arm_payload(arm, n=n, stop_reason=reason))
                loaded = analysis.load_arm_result(path, arm)
                self.assertEqual(loaded["observed_trajectory_length"], n)
                self.assertEqual(loaded["terminated_before_N254"], n < 254)
                self.assertFalse(loaded["unobserved_tail_extrapolated"])

    def test_direct_continuation_must_be_complete_and_hash_linked(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            original = root / "original.json"
            original.write_text("{}\n", encoding="utf-8")
            result = root / "direct.npz"
            np.savez(
                result,
                **synthetic_arm_payload(
                    "direct_lbfgs_only", n=3, stop_reason="cached_coordinate_stall"
                ),
            )
            report = root / "continuation.json"
            protocol = {
                "status": "preregistered",
                "protocol_id": analysis.DIRECT_CONTINUATION_PROTOCOL_ID,
                "continuation_of_protocol": analysis.PROTOCOL_ID,
                "cell_order": ["direct_lbfgs_only"],
                "single_new_cell": True,
                "scientific_command_exact_original_direct_clone": True,
                "scientific_command_only_routing_changed": True,
                "fixed_alpha": 15.0,
                "fixed_eta": 1.1,
                "fixed_residual_ratio": 0.8,
                "fixed_max_evaluated_steps": 10,
                "fixed_lbfgs_history": 7,
                "fixed_kkt_atol": 0.0,
                "budget_per_image": 254,
                "trajectory_record_count_allowed_range": [1, 254],
                "early_stop_counted_as_converged": False,
                "unreachable_threshold_policy": "candidate_unreachable_is_loss",
                "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
                "post_hoc_scientific_command_changes_allowed": False,
                "prior_artifact_manifest": {
                    "original_v1_report": {
                        "path": str(original),
                        "sha256": analysis.file_sha256(original),
                    }
                },
            }
            prereg = root / "continuation_preregistered.json"
            prereg.write_text(json.dumps(protocol), encoding="utf-8")
            payload = {
                **protocol,
                "status": "complete",
                "preregistration": str(prereg),
                "preregistration_sha256": analysis.file_sha256(prereg),
                "results": [
                    {
                        "arm": "direct_lbfgs_only",
                        "returncode": 0,
                        "result": str(result),
                        "result_sha256": analysis.file_sha256(result),
                        "trajectory_record_count": 3,
                        "trajectory_complete_to_budget": False,
                        "early_stop": True,
                        "precision_beyond_tightest_reached_unreachable": True,
                    }
                ],
            }
            report.write_text(json.dumps(payload), encoding="utf-8")
            _, extracted, expected_hash = analysis.extract_direct_continuation_result(
                report, original
            )
            self.assertEqual(extracted, result.resolve())
            self.assertEqual(expected_hash, analysis.file_sha256(result))
            payload["prior_artifact_manifest"]["original_v1_report"]["sha256"] = "0" * 64
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                analysis.extract_direct_continuation_result(report, original)

    def test_formal_analyze_refuses_to_run_before_direct_report_exists(self):
        with self.assertRaises(FileNotFoundError):
            analysis.analyze(Path("unused-original.json"), Path("missing-direct.json"))

    def test_npz_allowlists_and_source_are_objective_blind(self):
        forbidden = ("barrier", "densif", "potential_energy", "energies")
        for arm, keys in analysis.NPZ_ACCESS_ALLOWLISTS.items():
            self.assertFalse(
                any(any(token in key.lower() for token in forbidden) for key in keys),
                arm,
            )
        source = Path(analysis.__file__).read_text(encoding="utf-8")
        for quote in ('saved["', "saved['"):
            for token in forbidden:
                self.assertNotIn(f"{quote}{token}", source)

    def test_formal_real_inputs_cover_only_allowlist_and_give_registered_decision(self):
        accessed: list[str] = []
        real_load = np.load

        def tracking_load(*args, **kwargs):
            loaded = real_load(*args, **kwargs)
            if hasattr(loaded, "files"):
                return _TrackedNpz(loaded, accessed)
            return loaded

        with mock.patch.object(np, "load", side_effect=tracking_load):
            payload = analysis.analyze(self.original_report, self.direct_report)

        allowed = set().union(*analysis.NPZ_ACCESS_ALLOWLISTS.values())
        self.assertTrue(accessed)
        self.assertTrue(set(accessed).issubset(allowed), sorted(set(accessed) - allowed))
        json.dumps(payload, allow_nan=False)
        self.assertEqual(
            {
                arm: row["observed_trajectory_length"]
                for arm, row in payload["arms"].items()
            },
            {
                "current_ffrbk_selector": 254,
                "scalar_selector": 58,
                "direct_lbfgs_only": 133,
            },
        )
        mechanism = payload["mechanism_bidirectional_comparisons"]
        self.assertEqual(
            mechanism["ffrbk_operator_effect"]
            ["current_ffrbk_selector_vs_scalar_selector"]["W_T_L_U"],
            {"W": 5, "T": 1, "L": 0, "U": 0},
        )
        self.assertEqual(
            mechanism["selector_effect"]
            ["scalar_selector_vs_direct_lbfgs_only"]["W_T_L_U"],
            {"W": 2, "T": 1, "L": 3, "U": 3},
        )
        self.assertEqual(
            mechanism["selector_effect"]
            ["direct_lbfgs_only_vs_scalar_selector"]["W_T_L_U"],
            {"W": 1, "T": 1, "L": 4, "U": 0},
        )
        self.assertEqual(payload["performance_qualifiers"], [])
        self.assertFalse(payload["winner_claim_made"])
        self.assertFalse(payload["physical_objective_arrays_accessed"])
        self.assertFalse(payload["unobserved_tail_extrapolated"])


if __name__ == "__main__":
    unittest.main()
