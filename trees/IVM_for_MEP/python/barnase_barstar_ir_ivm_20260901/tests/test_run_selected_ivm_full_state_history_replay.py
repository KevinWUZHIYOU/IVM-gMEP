"""Zero-OpenMM tests for the selected full-state-history replay."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
for directory in (ROOT / "scripts", ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RECORDER = load(
    "bb_full_state_recorder_test", ROOT / "scripts/full_state_history_recorder.py"
)
RUNNER = load(
    "bb_full_state_runner_test",
    ROOT / "scripts/run_selected_ivm_full_state_history_replay.py",
)
VERIFIER = load(
    "bb_full_state_verifier_test",
    ROOT / "scripts/verify_selected_ivm_full_state_history_replay.py",
)


class FullStateHistoryReplayTest(unittest.TestCase):
    def setUp(self):
        registration = json.loads(
            (
                ROOT
                / "registrations/production/bb_irivm_a1e0_r01_m05_inf9990_r1.json"
            ).read_text(encoding="utf-8")
        )
        self.reference = [str(value) for value in registration["command"]]

    def test_runtime_delta_changes_only_two_output_values(self):
        replay = RUNNER.replay_argv({"command": self.reference})
        RUNNER.validate_runtime_delta(self.reference, replay)
        differences = [
            (left, right)
            for left, right in zip(self.reference, replay)
            if left != right
        ]
        self.assertEqual(
            differences,
            [
                ("runs/production", "runs/full_state_history_replay"),
                (
                    "bb_irivm_a1e0_r01_m05_inf9990_r1",
                    "bb_irivm_a1e0_r01_m05_inf9990_fullstate_r2",
                ),
            ],
        )
        self.assertEqual(RUNNER.PATH_ONLY.option_value(replay, "--path-storage-mode"), "committed")
        self.assertNotIn("--save-all-paths", replay)
        self.assertEqual(RUNNER.PATH_ONLY.option_value(replay, "--budget"), "9990")
        self.assertEqual(
            RUNNER.PATH_ONLY.option_value(replay, "--total-internal-nf-budget"),
            "9990",
        )

    def test_any_algorithm_or_storage_drift_is_rejected(self):
        for flag, value in (("--alpha", "10"), ("--path-storage-mode", "legacy")):
            with self.subTest(flag=flag):
                replay = RUNNER.replay_argv({"command": self.reference})
                replay[replay.index(flag) + 1] = value
                with self.assertRaises(RuntimeError):
                    RUNNER.validate_runtime_delta(self.reference, replay)

    def test_observer_writes_one_atomic_full_state_without_changing_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            registration = temporary_path / "registration.json"
            registration.write_text("{}\n", encoding="utf-8")
            original_calls = []

            def original_policy(**kwargs):
                original_calls.append(kwargs)
                return True

            fake_runner = SimpleNamespace(should_store_path=original_policy)
            recorder = RECORDER.FullStateHistoryRecorder(
                temporary_path / "observer",
                RUNNER.LABEL,
                registration,
                RECORDER.sha256_file(registration),
            )
            restore = RECORDER.install_output_observer(fake_runner, recorder)

            def record():
                path = np.zeros((32, 9396), dtype=np.float64)
                energies = np.arange(32, dtype=np.float64)
                forces = np.ones((32, 9396), dtype=np.float64)
                internal = np.zeros(32, dtype=np.int64)
                internal[1:-1] = 1
                endpoint_hits = np.zeros(32, dtype=np.int64)
                endpoint_hits[[0, -1]] = 1
                requests = np.ones(32, dtype=np.int64)
                counters = {
                    "joint_evaluations": internal,
                    "energy_requests": requests,
                    "force_requests": requests,
                    "cache_hits": endpoint_hits,
                    "total_query_events": requests,
                    "both_requests": requests,
                    "budget_blocked_misses": np.zeros(32, dtype=np.int64),
                    "inherited_cache_entries": endpoint_hits,
                }
                history = {
                    "calls_per_image": [1],
                    "total_internal_joint_evaluations": [30],
                }
                diagnostic = {"i_perp": 7.0}
                return fake_runner.should_store_path(
                    mode="committed",
                    save_all_paths=False,
                    call=1,
                    is_initial=True,
                    reparam_committed=False,
                )

            try:
                self.assertTrue(record())
            finally:
                restore()
            self.assertEqual(len(original_calls), 1)
            self.assertEqual(len(recorder.records), 1)
            shard = Path(recorder.records[0]["shard"])
            self.assertTrue(shard.is_file())
            self.assertFalse(shard.with_name(shard.name + ".tmp").exists())
            with np.load(shard, allow_pickle=False) as saved:
                self.assertEqual(saved["path32_A"].shape, (32, 9396))
                self.assertEqual(saved["energy_eV"].shape, (32,))
                self.assertEqual(saved["force_eV_per_A"].shape, (32, 9396))
                self.assertEqual(int(saved["additional_physical_calls"]), 0)
                self.assertEqual(int(saved["optimization_N_F"]), 30)
            events = [json.loads(line) for line in recorder.wal.read_text().splitlines()]
            self.assertEqual([event["event"] for event in events], ["header", "record_committed"])

    def test_freeze_authorize_and_preflight_are_zero_openmm(self):
        with tempfile.TemporaryDirectory() as temporary:
            registration_path = Path(temporary) / "registration.json"
            authorization_path = Path(temporary) / "authorization.json"
            RUNNER.freeze(ROOT, registration_path)
            RUNNER.authorize(ROOT, registration_path, authorization_path)
            registration = json.loads(registration_path.read_text(encoding="utf-8"))
            authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
            self.assertEqual(registration["status"], RUNNER.STATUS)
            self.assertEqual(
                registration["only_permitted_runtime_argv_differences"],
                {
                    "--run-label": RUNNER.LABEL,
                    "--output-dir": RUNNER.OUTPUT_SUBDIR,
                },
            )
            capture = registration["output_only_capture"]
            self.assertEqual(capture["authoritative_shard_calls_required"], list(range(1, 353)))
            self.assertEqual(capture["formally_reportable_call_range_after_exact_verification"], [1, 318])
            self.assertEqual(capture["excluded_terminal_uncommitted_tail_call_range"], [319, 352])
            self.assertEqual(registration["resource_estimate"]["minimum_free_disk_bytes_before_child"], 6 * 1024**3)
            self.assertEqual(registration["resource_estimate"]["exact_raw_path_energy_force_shard_payload_bytes"], 1693474816)
            selected = json.loads(
                (
                    ROOT
                    / "registrations/production/bb_irivm_a1e0_r01_m05_inf9990_r1.json"
                ).read_text(encoding="utf-8")
            )
            for name, source_hash in selected["source_sha256"].items():
                self.assertEqual(registration["source_sha256"][name], source_hash)
            self.assertTrue(authorization["enabled"])
            with mock.patch.dict(
                os.environ,
                {name: "1" for name in RUNNER.PATH_ONLY.THREAD_CAPS},
                clear=False,
            ), mock.patch.object(
                RUNNER.shutil,
                "disk_usage",
                return_value=SimpleNamespace(total=20 * 1024**3, used=10 * 1024**3, free=10 * 1024**3),
            ):
                RUNNER.preflight(ROOT, registration_path, authorization_path)

    def test_discrete_i_perp_matches_article_definition(self):
        coordinates = np.zeros((32, 9396), dtype=np.float64)
        coordinates[:, 0] = np.arange(32, dtype=np.float64)
        force = np.zeros_like(coordinates)
        force[1:-1, 1] = 2.0
        self.assertEqual(VERIFIER.discrete_i_perp(coordinates, force), 120.0)

    def test_capture_source_contains_no_evaluator_call(self):
        source = (ROOT / "scripts/full_state_history_recorder.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("evaluate_path(", source)
        self.assertNotIn(".query(", source)
        self.assertNotIn("getState(", source)


if __name__ == "__main__":
    unittest.main()
