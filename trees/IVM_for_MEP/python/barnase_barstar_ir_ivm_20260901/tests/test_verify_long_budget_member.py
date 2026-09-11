from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_long_budget_member as verifier  # noqa: E402


REFERENCE_990 = ROOT / (
    "runs/screen/"
    "ivm_bb_irivm_a1e1_r05_m05_inf990_r3_central_state_ivm_"
    "cfg2a53ad9cf3db_budget990_storagecommitted.npz"
)


class VerifyLongBudgetMemberTests(unittest.TestCase):
    def test_valid_986_atomic_remainder(self) -> None:
        with np.load(REFERENCE_990, allow_pickle=False) as saved:
            partial, quarantined, case = verifier.validate_budget_atomicity(
                saved, actual=986, budget=990
            )
        self.assertTrue(partial)
        self.assertTrue(quarantined)
        self.assertEqual(case, "atomic_remainder")

    def test_non_budget_early_stop_is_rejected(self) -> None:
        with np.load(REFERENCE_990, allow_pickle=False) as saved:
            fixture = {
                key: np.asarray(saved[key])
                for key in (
                    "total_internal_nf_terminal_partial_tail_present",
                    "total_internal_nf_terminal_partial_tail_quarantined",
                    "total_internal_nf_budget_stop_reason",
                    "total_internal_nf_full_band_preflight_zero_step_stops",
                    "total_internal_nf_complete_band_budget_stop_count",
                )
            }
        fixture["total_internal_nf_budget_stop_reason"] = np.asarray("converged")
        with self.assertRaisesRegex(RuntimeError, "non-budget stop reason"):
            verifier.validate_budget_atomicity(fixture, actual=986, budget=990)

    def test_post_initial_committed_paths_are_bound_to_nf_and_hash(self) -> None:
        with np.load(REFERENCE_990, allow_pickle=False) as saved:
            eligible, records = verifier.collect_committed_stored_paths(saved)
        self.assertEqual(eligible, [1, 3, 10, 32])
        self.assertEqual(
            [(item["call_per_image"], item["actual_optimization_N_F"]) for item in records],
            [(3, 90), (10, 300), (32, 716)],
        )
        self.assertTrue(all(len(item["path_sha256"]) == 64 for item in records))

    def test_initial_path_alone_is_not_sufficient(self) -> None:
        fixture = {
            "calls_per_image": np.asarray([1, 2]),
            "reparam_committed": np.asarray([False, False]),
            "stored_path_calls": np.asarray([1]),
            "stored_paths": np.zeros((1, 32, 3), dtype=np.float64),
            "counter_vectors": np.zeros((2, 32), dtype=np.int64),
        }
        with self.assertRaisesRegex(RuntimeError, "no post-initial"):
            verifier.collect_committed_stored_paths(fixture)


if __name__ == "__main__":
    unittest.main()
