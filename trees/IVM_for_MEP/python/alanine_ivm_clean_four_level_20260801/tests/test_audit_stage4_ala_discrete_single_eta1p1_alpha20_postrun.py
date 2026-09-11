from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import audit_stage4_ala_discrete_single_eta1p1_alpha15_postrun as alpha15_engine
import audit_stage4_ala_discrete_single_eta1p1_alpha20_postrun as audit


class Eta1p1Alpha20PostrunAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.report = cls.root / audit.FORMAL_REPORT_RELATIVE

    def test_completed_formal_run_passes_independent_audit(self):
        payload = audit.build_audit(self.root, self.report)
        self.assertEqual(payload["status"], "passed")
        self.assertTrue(payload["read_only_postrun_audit"])
        self.assertFalse(payload["barrier_metric_values_accessed"])
        self.assertFalse(payload["energy_or_objective_fallback_used"])
        provenance = payload["provenance"]
        self.assertEqual(provenance["run_source_count"], 16)
        self.assertEqual(provenance["decision_input_count"], 13)
        self.assertTrue(provenance["all_artifact_cross_links_verified"])
        self.assertTrue(
            provenance["scientific_command_contract"][
                "alpha_is_only_scientific_argv_difference_from_alpha15"
            ]
        )
        self.assertEqual(
            [
                row["index"]
                for row in provenance["scientific_command_contract"][
                    "alpha15_alpha20_difference_records"
                ]
            ],
            [7, 9, 13],
        )
        mechanism = payload["mechanism_recomputation"]
        self.assertEqual(mechanism["joint_selected_step_count"], 81)
        self.assertEqual(mechanism["h0_selection_count"], 44)
        self.assertEqual(mechanism["lbfgs_selection_count"], 37)
        self.assertEqual(mechanism["cap_hit_count"], 1)
        self.assertEqual(mechanism["raw_kmax_stop_count"], 1)
        self.assertEqual(mechanism["final_optimization_NF"], 3370)
        self.assertEqual(mechanism["subproblem_line_search_backtrack_count"], 0)
        self.assertTrue(mechanism["alpha_exactly_20_throughout"])
        self.assertTrue(mechanism["fixed_r08_k10_m7_verified"])
        failure_keys = [
            key
            for key in mechanism
            if key.endswith("_mismatch_count")
            or key
            in {
                "base_exceeds_dynamic_cap_count",
                "applied_below_base_count",
                "nonincreasing_total_nf_count",
            }
        ]
        for key in failure_keys:
            self.assertEqual(mechanism[key], 0, key)

    def test_alpha15_invariant_engine_globals_are_restored(self):
        before = {
            name: getattr(alpha15_engine, name)
            for name in (
                "ALPHA",
                "CONFIGURATION_SHA256",
                "RESULT_RELATIVE",
                "MEMBER_AUDIT_RELATIVE",
                "COMPANION_AUDIT_RELATIVE",
            )
        }
        audit.audit_mechanism(self.root / audit.RESULT_RELATIVE)
        after = {name: getattr(alpha15_engine, name) for name in before}
        self.assertEqual(before, after)

    @staticmethod
    def _synthetic_mechanism() -> dict[str, np.ndarray]:
        selection = np.asarray(["H0", "LBFGS"], dtype="U8")
        base = np.asarray([0.2, 0.8])
        dynamic = np.asarray([1.0, 0.85])
        requested = audit.ETA * base
        applied = np.minimum(requested, dynamic)
        budget = audit.BUDGET
        total_nf = np.arange(1, budget + 1) * 5
        counter = np.zeros((budget, 22), dtype=int)
        counter[:, 0] = 1
        counter[:, -1] = 1
        counter[:, 1] = total_nf
        cache = np.zeros((budget, 22), dtype=int)
        cache[:, 0] = 1
        cache[:, -1] = 1
        requests = counter + cache
        return {
            "configuration_sha256": np.asarray(audit.CONFIGURATION_SHA256),
            "initial_path_sha256": np.asarray(audit.INITIAL_PATH_SHA256),
            "alpha": np.asarray(audit.ALPHA),
            "alpha_mode": np.asarray("fixed"),
            "active_alpha": np.full(budget, audit.ALPHA),
            "alpha_action": np.asarray(["initial"] + ["fixed"] * (budget - 1)),
            "budget": np.asarray(budget),
            "frozen_subproblem": np.asarray("discrete_b040"),
            "discrete_stop_mode": np.asarray("r08c10"),
            "discrete_inner_max_steps": np.asarray(audit.MAX_EVALUATED_STEPS),
            "discrete_residual_ratio": np.asarray(audit.RESIDUAL_RATIO),
            "discrete_history": np.asarray(audit.HISTORY),
            "discrete_audit_history_count": np.asarray([1, 2]),
            "discrete_audit_outer": np.asarray([0, 0]),
            "discrete_audit_step": np.asarray([0, 1]),
            "discrete_audit_stop_reason": np.asarray(
                ["evaluated-relative-natural-residual"]
            ),
            "discrete_audit_global_relative_residual": np.asarray([0.7]),
            "discrete_audit_selection": selection,
            "discrete_audit_h0_capped_predicted_decrease": np.asarray([2.0, 1.0]),
            "discrete_audit_lbfgs_capped_predicted_decrease": np.asarray([1.0, 2.0]),
            "discrete_h0_selection_count": np.asarray(1),
            "discrete_lbfgs_selection_count": np.asarray(1),
            "discrete_step_multiplier_eta": np.asarray(audit.ETA),
            "discrete_step_multiplier_selector_isolated": np.asarray(True),
            "discrete_step_multiplier_dynamic_cap_reapplied": np.asarray(True),
            "discrete_step_multiplier_energy_backtracking": np.asarray(False),
            "discrete_step_multiplier_extra_physical_evaluations": np.asarray(0),
            "discrete_step_multiplier_audit_selection": selection.copy(),
            "discrete_step_multiplier_audit_requested": np.full(2, audit.ETA),
            "discrete_step_multiplier_audit_applied": applied / base,
            "discrete_step_multiplier_audit_cap_hit": requested > dynamic,
            "discrete_step_multiplier_audit_dynamic_max_step": dynamic,
            "discrete_step_multiplier_audit_base_max_atomic_displacement": base,
            "discrete_step_multiplier_audit_requested_max_atomic_displacement": requested,
            "discrete_step_multiplier_audit_applied_max_atomic_displacement": applied,
            "discrete_step_multiplier_uncommitted_selection_count": np.asarray(0),
            "discrete_step_multiplier_aborted_selection_count": np.asarray(0),
            "subproblem_line_search_backtracks": np.zeros(budget, dtype=int),
            "reparam_mode": np.asarray("state_curve"),
            "state_curve_direct_objective_acceptance": np.asarray("per_image"),
            "state_curve_projection_uses_spline": np.asarray(False),
            "geometric_reparam_joint_evaluations": np.asarray(0),
            "stored_path_calls": np.arange(1, budget + 1),
            "total_internal_joint_evaluations": total_nf,
            "counter_vectors": counter,
            "cache_hit_vectors": cache,
            "energy_request_vectors": requests,
            "force_request_vectors": requests.copy(),
            # Object arrays fail under allow_pickle=False if a barrier is indexed.
            "optimization_barrier_is_sparse": np.asarray(["embargoed"], dtype=object),
            "barrier_metric_for_figures": np.asarray(["embargoed"], dtype=object),
            "barrier_kcal": np.asarray(["embargoed"], dtype=object),
        }

    def _write_synthetic(self, directory: Path, **updates: np.ndarray) -> Path:
        payload = self._synthetic_mechanism()
        payload.update(updates)
        target = directory / "synthetic.npz"
        np.savez(target, **payload)
        return target

    def test_synthetic_barriers_are_not_indexed_and_cap_is_recomputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self._write_synthetic(Path(temporary))
            record = audit.audit_mechanism(result)
        self.assertEqual(record["selector_winner_mismatch_count"], 0)
        self.assertEqual(record["cap_hit_count"], 1)
        self.assertEqual(
            record["applied_min_eta_base_dynamic_cap_mismatch_count"], 0
        )
        self.assertFalse(record["barrier_metric_values_accessed"])
        self.assertFalse(
            any("barrier" in key.lower() for key in record["mechanism_fields_accessed"])
        )

    def test_rejects_selector_recap_alpha_and_backtrack_drift(self):
        cases = {
            "selector": {
                "discrete_audit_selection": np.asarray(["LBFGS", "LBFGS"]),
                "discrete_step_multiplier_audit_selection": np.asarray(
                    ["LBFGS", "LBFGS"]
                ),
                "discrete_h0_selection_count": np.asarray(0),
                "discrete_lbfgs_selection_count": np.asarray(2),
            },
            "recap": {
                "discrete_step_multiplier_audit_applied_max_atomic_displacement": np.asarray(
                    [0.3, 0.85]
                ),
                "discrete_step_multiplier_audit_applied": np.asarray(
                    [1.5, 0.85 / 0.8]
                ),
            },
            "alpha": {"active_alpha": np.full(audit.BUDGET, 15.0)},
            "backtrack": {
                "subproblem_line_search_backtracks": np.asarray(
                    [1] + [0] * (audit.BUDGET - 1)
                )
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            for name, updates in cases.items():
                case_dir = temporary / name
                case_dir.mkdir()
                result = self._write_synthetic(case_dir, **updates)
                with self.subTest(name=name), self.assertRaises(RuntimeError):
                    audit.audit_mechanism(result)

    def test_scientific_command_validator_rejects_control_drift(self):
        prereg = json.loads(
            (self.root / audit.PREREGISTRATION_RELATIVE).read_text(encoding="utf-8")
        )
        command = json.loads(
            (self.root / audit.MEMBER_COMMAND_RELATIVE).read_text(encoding="utf-8")
        )
        command[command.index("--history") + 1] = "6"
        with self.assertRaisesRegex(RuntimeError, "--history"):
            audit._verify_scientific_command(
                self.root, command, prereg["decision_inputs"]
            )

    def test_sidecar_write_is_exclusive_and_strict_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "sidecar.json"
            audit.write_exclusive_strict(target, {"status": "passed", "x": 1.0})
            self.assertEqual(json.loads(target.read_text())["status"], "passed")
            with self.assertRaises(FileExistsError):
                audit.write_exclusive_strict(target, {"status": "overwrite"})
            with self.assertRaises(ValueError):
                audit.write_exclusive_strict(
                    Path(temporary) / "invalid.json", {"x": float("nan")}
                )


if __name__ == "__main__":
    unittest.main()
