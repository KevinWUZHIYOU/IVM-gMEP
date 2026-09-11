"""Zero-OpenMM tests for the selected full-history replay controls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = load("bb_full_history_runner_test", ROOT / "scripts/run_selected_ivm_full_history_replay.py")
VERIFIER = load("bb_full_history_verifier_test", ROOT / "scripts/verify_selected_ivm_full_history_replay.py")


class FullHistoryReplayTest(unittest.TestCase):
    def setUp(self):
        registration = json.loads((ROOT / "registrations/production/bb_irivm_a1e0_r01_m05_inf9990_r1.json").read_text())
        self.reference = [str(value) for value in registration["command"]]

    def test_runtime_delta_is_output_only_and_budget_unchanged(self):
        replay = RUNNER.replay_argv({"command": self.reference})
        RUNNER.validate_runtime_delta(self.reference, replay)
        reduced = replay.copy()
        reduced.remove("--save-all-paths")
        differences = [(left, right) for left, right in zip(self.reference, reduced) if left != right]
        self.assertEqual(
            differences,
            [
                ("runs/production", "runs/full_history_replay"),
                ("bb_irivm_a1e0_r01_m05_inf9990_r1", "bb_irivm_a1e0_r01_m05_inf9990_fullhistory_r1"),
                ("committed", "legacy"),
            ],
        )
        self.assertEqual(RUNNER.option_value(replay, "--budget"), "9990")
        self.assertEqual(RUNNER.option_value(replay, "--total-internal-nf-budget"), "9990")

    def test_algorithm_drift_is_rejected(self):
        replay = RUNNER.replay_argv({"command": self.reference})
        replay[replay.index("--alpha") + 1] = "10"
        with self.assertRaises(RuntimeError):
            RUNNER.validate_runtime_delta(self.reference, replay)

    def test_all_path_retention_does_not_change_online_digest(self):
        from storage_policy import SyntheticPrefixRecorder
        paths = [np.asarray([float(call), float(call + 1)]) for call in range(1, 353)]
        full = SyntheticPrefixRecorder(mode="legacy", save_all_paths=True)
        thin = SyntheticPrefixRecorder(mode="legacy", save_all_paths=False)
        for call, path in enumerate(paths, start=1):
            full.observe(call, path, reparam_committed=call in {7, 20, 72, 318})
            thin.observe(call, path, reparam_committed=call in {7, 20, 72, 318})
        self.assertEqual(full.online_digest, thin.online_digest)
        self.assertEqual([call for call, _path in full.stored], list(range(1, 353)))

    def test_freeze_and_authorize_are_zero_openmm(self):
        with tempfile.TemporaryDirectory() as temporary:
            registration_path = Path(temporary) / "registration.json"
            authorization_path = Path(temporary) / "authorization.json"
            RUNNER.freeze(ROOT, registration_path)
            RUNNER.authorize(ROOT, registration_path, authorization_path)
            registration = json.loads(registration_path.read_text())
            authorization = json.loads(authorization_path.read_text())
            self.assertEqual(registration["status"], RUNNER.STATUS)
            self.assertEqual(registration["path_retention"]["stored_calls_required"], list(range(1, 353)))
            self.assertEqual(registration["path_retention"]["formally_reportable_call_range_after_exact_verification"], [1, 318])
            self.assertEqual(registration["path_retention"]["excluded_terminal_uncommitted_tail_call_range"], [319, 352])
            self.assertTrue(authorization["enabled"])
            self.assertEqual(authorization["minimum_free_disk_bytes_before_child"], 12 * 1024**3)

    def test_inventory_marks_tail_strictly(self):
        calls = np.arange(1, 353)
        nf = np.arange(1, 353) * 30
        values = np.arange(1, 353, dtype=float)
        committed = np.zeros(352, dtype=bool)
        paths = np.zeros((352, 1), dtype=float)
        inventory = VERIFIER.classify_inventory(calls, nf, values, committed, paths)
        self.assertTrue(inventory[317]["formally_reportable"])
        self.assertFalse(inventory[318]["formally_reportable"])
        self.assertTrue(all(row["terminal_uncommitted_charged_tail"] for row in inventory[318:]))


if __name__ == "__main__":
    unittest.main()
