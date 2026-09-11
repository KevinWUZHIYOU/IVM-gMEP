"""Zero-OpenMM gates for the native all-path selected IR-IVM replay."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = load(
    "bb_all_paths_r2_runner_test",
    ROOT / "scripts/run_selected_ivm_all_paths_replay_r2.py",
)
VERIFIER = load(
    "bb_all_paths_r2_verifier_test",
    ROOT / "scripts/verify_selected_ivm_all_paths_replay_r2.py",
)


class AllPathsReplayR2Test(unittest.TestCase):
    def setUp(self):
        selected = json.loads(
            (
                ROOT
                / "registrations/production/bb_irivm_a1e0_r01_m05_inf9990_r1.json"
            ).read_text(encoding="utf-8")
        )
        self.reference = [str(value) for value in selected["command"]]

    def test_runtime_delta_is_storage_only_and_budget_is_9990(self):
        replay = RUNNER.replay_argv({"command": self.reference})
        RUNNER.validate_runtime_delta(self.reference, replay)
        reduced = replay.copy()
        reduced.remove("--save-all-paths")
        differences = [
            (left, right)
            for left, right in zip(self.reference, reduced)
            if left != right
        ]
        self.assertEqual(
            differences,
            [
                ("runs/production", "runs/all_paths_replay_r2"),
                (
                    "bb_irivm_a1e0_r01_m05_inf9990_r1",
                    "bb_irivm_a1e0_r01_m05_inf9990_allpaths_r2",
                ),
                ("committed", "legacy"),
            ],
        )
        self.assertEqual(RUNNER.BASE.option_value(replay, "--budget"), "9990")
        self.assertEqual(
            RUNNER.BASE.option_value(replay, "--total-internal-nf-budget"),
            "9990",
        )

    def test_algorithm_change_is_rejected(self):
        replay = RUNNER.replay_argv({"command": self.reference})
        replay[replay.index("--alpha") + 1] = "10"
        with self.assertRaises(RuntimeError):
            RUNNER.validate_runtime_delta(self.reference, replay)

    def test_native_storage_decision_does_not_change_online_state(self):
        from storage_policy import SyntheticPrefixRecorder

        calls = [1, 2, 3, 4, 5, 5, 6, 7, 7, 8]
        full = SyntheticPrefixRecorder(mode="legacy", save_all_paths=True)
        thin = SyntheticPrefixRecorder(mode="legacy", save_all_paths=False)
        for ordinal, call in enumerate(calls):
            path = np.asarray([ordinal, call], dtype=np.float64)
            committed = call in {7}
            self.assertTrue(
                np.array_equal(
                    full.observe(call, path, reparam_committed=committed),
                    thin.observe(call, path, reparam_committed=committed),
                )
            )
        self.assertEqual(full.online_digest, thin.online_digest)
        self.assertEqual(len(full.stored), len(calls))

    def test_successful_log_and_production_finalize_to_352_last_wins_paths(self):
        log = (
            ROOT / "runs/logs/bb_irivm_a1e0_r01_m05_inf9990_r1.log"
        ).read_text(encoding="utf-8")
        pattern = re.compile(r"^\[IVM central [^]]+\] calls=\s*(\d+)")
        raw_calls = [
            int(match.group(1))
            for line in log.splitlines()
            if (match := pattern.match(line))
        ]
        self.assertEqual(len(raw_calls), 455)
        calls = np.asarray(raw_calls, dtype=np.int64)
        _, reverse_indices = np.unique(calls[::-1], return_index=True)
        keep = len(calls) - 1 - reverse_indices
        keep = keep[np.argsort(calls[keep], kind="stable")]
        self.assertTrue(np.array_equal(calls[keep], np.arange(1, 353)))
        source = (
            ROOT / "src/run_ivm_v9_spring_projected_production_20260730.py"
        ).read_text(encoding="utf-8")
        self.assertIn("history_calls[::-1], return_index=True", source)
        self.assertIn("stored_call_labels[::-1], return_index=True", source)

    def test_freeze_authorize_and_preflight_are_zero_openmm(self):
        with tempfile.TemporaryDirectory() as temporary:
            registration_path = Path(temporary) / "registration.json"
            authorization_path = Path(temporary) / "authorization.json"
            RUNNER.freeze(ROOT, registration_path)
            RUNNER.authorize(ROOT, registration_path, authorization_path)
            registration = json.loads(registration_path.read_text())
            authorization = json.loads(authorization_path.read_text())
            self.assertEqual(registration["schema"], RUNNER.SCHEMA)
            self.assertEqual(registration["status"], RUNNER.STATUS)
            self.assertFalse(registration["full_state_observer_installed"])
            self.assertEqual(
                registration["path_retention"]["stored_calls_required"],
                list(range(1, 353)),
            )
            self.assertEqual(
                authorization["minimum_free_disk_bytes_before_child"],
                6 * 1024**3,
            )
            caps = {name: "1" for name in RUNNER.BASE.THREAD_CAPS}
            with mock.patch.dict(os.environ, caps, clear=False):
                RUNNER.preflight(ROOT, registration_path, authorization_path)

    def test_verifier_wrapper_identity(self):
        self.assertEqual(
            VERIFIER.SCHEMA,
            "bb-irivm-selected-all-paths-replay-verification-v2",
        )
        self.assertEqual(VERIFIER.LABEL, RUNNER.LABEL)


if __name__ == "__main__":
    unittest.main()
