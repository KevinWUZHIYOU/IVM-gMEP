from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_selected_9990_prefix as verifier  # noqa: E402


class ArchiveFixture:
    def __init__(self, arrays):
        self.arrays = arrays
        self.files = list(arrays)

    def __getitem__(self, key):
        return self.arrays[key]


def archive(extra=False):
    count = 6 if extra else 4
    calls = np.arange(1, count + 1, dtype=np.int64)
    committed = np.asarray(
        [False, True, False, True] + ([False, True] if extra else []),
        dtype=bool,
    )
    counters = np.zeros((count, 32), dtype=np.int64)
    for index in range(count):
        counters[index, 1:-1] = index + 1
    history = np.asarray(
        [8.0, 6.0, 4.0, 2.0] + ([1.5, 1.0] if extra else []),
        dtype=np.float64,
    )
    path_by_call = {
        call: np.full((32, 3), float(call), dtype=np.float64)
        for call in calls[committed]
    }
    stored_calls = [1, 2, 4] + ([6] if extra else [])
    stored_paths = [np.zeros((32, 3), dtype=np.float64)] + [
        path_by_call[call] for call in stored_calls[1:]
    ]
    audit_outer = np.asarray([0, 0, 1, 2], dtype=np.int64)
    return ArchiveFixture(
        {
            "calls_per_image": calls,
            "reparam_committed": committed,
            "counter_vectors": counters,
            "i_perp": history,
            "stored_path_calls": np.asarray(stored_calls, dtype=np.int64),
            "stored_paths": np.asarray(stored_paths, dtype=np.float64),
            "quality": np.arange(count, dtype=np.float64),
            "wall_seconds": np.arange(count, dtype=np.float64),
            "discrete_audit_outer": audit_outer,
            "discrete_audit_step": np.asarray([0, 1, 0, 0], dtype=np.int64),
        }
    )


class ExactPrefixTests(unittest.TestCase):
    def test_reference_is_exact_prefix_of_longer_trajectory(self):
        result = verifier.compare_exact_prefix(archive(), archive(extra=True))
        self.assertEqual(result["compared_online_history_record_count"], 4)
        self.assertEqual(result["selected_total_eligible_committed_state_count"], 3)
        self.assertNotIn("wall_seconds", result["compared_online_history_keys"])

    def test_path_mismatch_is_rejected(self):
        selected = archive(extra=True)
        selected.arrays["stored_paths"][1, 1, 1] += 1.0
        with self.assertRaisesRegex(RuntimeError, "stored path mismatch"):
            verifier.compare_exact_prefix(archive(), selected)

    def test_online_history_mismatch_is_rejected(self):
        selected = archive(extra=True)
        selected.arrays["quality"][2] += 1.0
        with self.assertRaisesRegex(RuntimeError, "online history prefix mismatch"):
            verifier.compare_exact_prefix(archive(), selected)


if __name__ == "__main__":
    unittest.main()
