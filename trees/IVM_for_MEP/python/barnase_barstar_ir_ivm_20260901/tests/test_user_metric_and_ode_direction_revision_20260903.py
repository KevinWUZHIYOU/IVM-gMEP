import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = (
    ROOT
    / "registrations/diagnostics/BB_USER_METRIC_AND_ODE_DIRECTION_REVISION_20260903_R1.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UserDirectionRevisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(REGISTRATION.read_text(encoding="utf-8"))

    def test_metric_and_experiment_direction_is_exact(self):
        p = self.payload
        self.assertEqual(p["schema"], "bb-user-metric-and-ode-direction-revision-v1")
        self.assertTrue(p["not_execution_authority"])
        self.assertEqual(
            p["experiment_direction"]["ode_screen"]["candidate_spring_k_eV_per_A2"],
            [0.1, 1.0, 10.0, 100.0],
        )
        self.assertEqual(p["experiment_direction"]["ode_screen"]["screen_budget_optimization_N_F_per_candidate"], 6000)
        self.assertEqual(p["experiment_direction"]["ode_screen"]["fresh_winner_budget_optimization_N_F"], 30000)
        self.assertTrue(p["experiment_direction"]["ode_screen"]["fresh_winner_must_restart_from_common_initial_path"])
        self.assertEqual(p["experiment_direction"]["storage"]["each_complete_force_batch"], ["path32_A", "energy_eV", "force_eV_per_A"])
        self.assertEqual(p["experiment_direction"]["storage"]["endpoint_optimization_N_F_contribution"], 0)
        self.assertEqual(p["experiment_direction"]["tangent"]["definition"], "tau_i = normalize(x[i+1] - x[i-1])")
        self.assertTrue(p["metric_policy"]["same_path_required_for_both_metrics"])
        self.assertIn("original 32-image path", p["metric_policy"]["DeltaE32"])
        self.assertIn("central-tangent discrete I_perp", p["metric_policy"]["I_perp"])
        self.assertEqual(p["metric_policy"]["arc_length_densification"], "forbidden_for_new_formal_figures")
        self.assertEqual(p["metric_policy"]["metric_interpolation"], "none")

    def test_referenced_existing_files_are_hash_bound(self):
        for role, binding in self.payload["referenced_existing_files"].items():
            with self.subTest(role=role):
                path = Path(binding["path"])
                self.assertTrue(path.is_file())
                self.assertEqual(sha256_file(path), binding["sha256"])

    def test_cancelled_formal_arc500_outputs_remain_absent(self):
        cancelled = [
            ROOT / "registrations/diagnostics/BB_IRIVM_COST_ONLY_LATE_SIX_SELECTION_NF15000_20260902_R3.json",
            ROOT / "registrations/diagnostics/BB_IRIVM_COST_LATE_SIX_LOCAL2_ARC500_CONTRACT_20260902_R3.json",
            ROOT / "registrations/diagnostics/BB_IRIVM_COST_LATE_SIX_REMOTE4_ARC500_CONTRACT_20260902_R3.json",
            ROOT / "registrations/execution_authorizations/bb_irivm_cost_late_six_local2_arc500_20260902_r3.json",
            ROOT / "registrations/execution_authorizations/bb_irivm_cost_late_six_remote4_arc500_20260902_r3.json",
            ROOT / "runs/diagnostics/bb_irivm_cost_late_six_local2_arc500_joint_ef_20260902_r3.npz",
            ROOT / "runs/diagnostics/bb_irivm_cost_late_six_remote4_arc500_joint_ef_20260902_r3.npz",
            ROOT / "registrations/diagnostics/BB_COST_TO_PRECISION_FINAL_IVM10_ODE8_R5B_20260902_R1.json",
        ]
        for path in cancelled:
            with self.subTest(path=path.name):
                self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
