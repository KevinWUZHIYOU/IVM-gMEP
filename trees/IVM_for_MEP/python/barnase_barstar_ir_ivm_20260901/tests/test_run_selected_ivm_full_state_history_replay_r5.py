"""Critical zero-physical tests for dynamic stable-last full-state R5B."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
import sys
for directory in (ROOT / "scripts", ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import full_state_history_recorder_r5 as RECORDER
import run_selected_ivm_full_state_history_replay_r5 as RUNNER
import verify_selected_ivm_full_state_history_replay_r5 as VERIFIER


def counters(call: int):
    joint = np.zeros(32, dtype=np.int64); joint[1:-1] = call
    endpoint = np.zeros(32, dtype=np.int64); endpoint[[0, -1]] = 1
    requests = joint + endpoint
    return {
        "joint_evaluations": joint,
        "energy_requests": requests.copy(),
        "force_requests": requests.copy(),
        "cache_hits": endpoint.copy(),
        "total_query_events": requests.copy(),
        "both_requests": requests.copy(),
        "budget_blocked_misses": np.zeros(32, dtype=np.int64),
        "inherited_cache_entries": endpoint.copy(),
    }


def state(marker: float):
    path = np.full((32, 9396), marker, dtype=np.float64)
    energy = np.arange(32, dtype=np.float64) + marker
    force = np.full((32, 9396), -marker, dtype=np.float64)
    return path, energy, force


class R5Test(unittest.TestCase):
    def recorder(self, directory: Path):
        registration = directory / "registration.json"; registration.write_text("{}\n")
        return RECORDER.FullStateHistoryRecorder(directory / "observer", "test-r5b", registration, RECORDER.sha256_file(registration))

    def record(self, recorder, history, call, marker):
        path, energy, force = state(marker)
        history["calls_per_image"].append(call)
        history["total_internal_joint_evaluations"].append(30 * call)
        recorder.record(call, path, energy, force, counters(call), history, {"i_perp": marker}, counter_source_kind="live_counter_arrays")
        return path

    def test_duplicate_calls_are_pending_last_wins_and_only_K_large_shards_remain(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); recorder = self.recorder(directory)
            history = {"calls_per_image": [], "total_internal_joint_evaluations": []}
            stable = {}
            for call, marker in ((1, 1.0), (2, 2.0), (2, 2.5), (3, 3.0), (3, 3.5)):
                stable[call] = self.record(recorder, history, call, marker)
            result = directory / "result.npz"; reference = directory / "reference.npz"
            np.savez(result, calls_per_image=np.arange(1, 4), total_internal_joint_evaluations=np.asarray([30, 60, 90]), stored_path_calls=np.arange(1, 4), stored_paths=np.asarray([stable[i] for i in range(1, 4)]), total_internal_nf_last_complete_state_index=np.asarray(2))
            # Synthetic call 3 models old call 352: it is outside the old
            # complete prefix and must not be required equal after extending
            # the budget.
            old_tail = stable[3].copy(); old_tail[0, 0] += 99.0
            np.savez(reference, stored_path_calls=np.asarray([1, 3]), stored_paths=np.asarray([stable[1], old_tail]))
            with mock.patch.object(RECORDER, "MAXIMUM_RAW_CALLBACKS", 20), mock.patch.object(RECORDER, "MAXIMUM_CANONICAL_CALLS", 10), mock.patch.object(RECORDER, "MINIMUM_RUNTIME_FREE_BYTES", 0), mock.patch.object(RECORDER, "OLD_COMPLETE_PREFIX_CALLS", 2):
                manifest, canonical, audit = recorder.finalize(result, reference)
            self.assertEqual(len(list(recorder.shards.glob("*.npz"))), 3)
            index = json.loads(canonical.read_text())
            self.assertEqual(index["raw_record_count"], 5)
            self.assertEqual(index["canonical_record_count"], 3)
            self.assertEqual(index["source_raw_record_ordinals"], [0, 2, 4])
            self.assertFalse((recorder.root / "pending_current_call.npz").exists())
            self.assertTrue(manifest.is_file() and audit.is_file())
            with np.load(recorder.shards / "call_0002_stable_last.npz", allow_pickle=False) as saved:
                self.assertEqual(float(saved["path32_A"][0, 0]), 2.5)

    def test_wrong_call_sequence_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = self.recorder(Path(temporary)); history = {"calls_per_image": [], "total_internal_joint_evaluations": []}
            with mock.patch.object(RECORDER, "MINIMUM_RUNTIME_FREE_BYTES", 0):
                self.record(recorder, history, 1, 1.0)
                with self.assertRaisesRegex(RuntimeError, "skipped"):
                    self.record(recorder, history, 3, 3.0)

    def test_observer_preserves_original_return_and_issues_no_query(self):
        calls = []
        class Sink:
            def record(self, *args, **kwargs): calls.append((args, kwargs))
        def should_store_path(**_kwargs): return "original-return"
        should_store_path.__name__ = "should_store_path"; should_store_path.__module__ = "storage_policy"
        fake = None
        def run():
            path, energies, forces = state(1.0); diagnostic = {"i_perp": 1.0}; history = {"calls_per_image": [], "total_internal_joint_evaluations": []}; live = counters(1)
            def record(counter_snapshot=None):
                counters_local = live if counter_snapshot is None else counter_snapshot
                counters_alias = counters_local
                history["calls_per_image"].append(1); history["total_internal_joint_evaluations"].append(30)
                counters = counters_alias
                _ = path, diagnostic
                return fake.should_store_path(mode="legacy", save_all_paths=True, call=1, is_initial=True, reparam_committed=False)
            def attempt_reparameterization(): return record()
            _ = attempt_reparameterization
            return record()
        fake = SimpleNamespace(__file__=str(Path(__file__).resolve()), run=run, should_store_path=should_store_path)
        restore = RECORDER.install_output_observer(fake, Sink(), expected_source_sha256=RECORDER.sha256_file(Path(__file__).resolve()))
        try: result = run()
        finally: restore()
        self.assertEqual(result, "original-return"); self.assertEqual(len(calls), 1)

    def test_runtime_delta_is_only_identity_plus_user_budget_extension(self):
        r2 = json.loads((ROOT / RUNNER.R2_REGISTRATION).read_text())
        reference = [str(value) for value in r2["runtime_argv"]]; replay = RUNNER.replay_argv(r2)
        RUNNER.validate_runtime_delta(reference, replay)
        differences = [(reference[i], replay[i]) for i in range(len(reference)) if reference[i] != replay[i]]
        self.assertEqual(len(differences), 4)
        self.assertEqual(RUNNER.BASE.option_value(replay, "--budget"), "15000")
        self.assertEqual(RUNNER.BASE.option_value(replay, "--total-internal-nf-budget"), "15000")
        self.assertEqual(RUNNER.BASE.option_value(replay, "--path-storage-mode"), "legacy")
        self.assertIn("--save-all-paths", replay)

    def test_verifier_requires_exact_1_to_318_prefix_but_not_old_tail(self):
        prefix = RECORDER.OLD_COMPLETE_PREFIX_CALLS
        reference = {"x": np.arange(prefix + 34, dtype=np.int64)}
        replay = {"x": np.arange(prefix + 202, dtype=np.int64)}
        replay["x"][prefix + 33] += 999
        VERIFIER.exact_prefix(reference, replay, "x")
        replay["x"][prefix - 1] += 1
        with self.assertRaisesRegex(RuntimeError, "prefix mismatch"):
            VERIFIER.exact_prefix(reference, replay, "x")

    def test_mutagen_template_has_real_newline_and_gate_roundtrip(self):
        self.assertIn("\n", RUNNER.MUTAGEN_LIST_TEMPLATE)
        self.assertNotIn("\\n", RUNNER.MUTAGEN_LIST_TEMPLATE)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            registration = directory / "registration.json"
            authorization = directory / "authorization.json"
            gate = directory / "mutagen.json"
            with mock.patch.object(RUNNER, "CANONICAL_REGISTRATION", registration), mock.patch.object(RUNNER, "CANONICAL_AUTHORIZATION", authorization), mock.patch.object(RUNNER, "CANONICAL_MUTAGEN_GATE", gate):
                RUNNER.freeze(ROOT, registration)
                RUNNER.authorize(ROOT, registration, authorization)

                def fake_run(argv, **_kwargs):
                    if argv[1:3] == ["sync", "flush"]:
                        return SimpleNamespace(returncode=0, stdout="", stderr="")
                    if "--template" in argv:
                        self.assertEqual(argv[-1], RUNNER.MUTAGEN_LIST_TEMPLATE)
                        return SimpleNamespace(returncode=0, stdout=f"{RUNNER.MUTAGEN_NAME}|{RUNNER.MUTAGEN_IDENTIFIER}|Watching|true|true\n", stderr="")
                    return SimpleNamespace(returncode=0, stdout="Status: Watching\n", stderr="")

                with mock.patch.object(RUNNER.subprocess, "run", side_effect=fake_run):
                    RUNNER.flush_mutagen(registration, authorization, gate)
                validated = RUNNER.validate_mutagen(registration, authorization, gate)
            self.assertEqual(validated["gate_generator"], RUNNER.BASE.bind(Path(RUNNER.__file__).resolve()))

    def test_freeze_authorize_and_remote_preflight_are_zero_openmm(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); registration = directory / "registration.json"; authorization = directory / "authorization.json"; gate = directory / "mutagen.json"
            # Keep the zero-OpenMM preflight isolated from a live canonical R5B
            # observer.  The overwrite guard is production behavior; this test
            # must exercise it against an empty temporary output directory.
            output_subdir = str(directory / "isolated-r5b-output")
            with mock.patch.object(RUNNER, "CANONICAL_REGISTRATION", registration), mock.patch.object(RUNNER, "CANONICAL_AUTHORIZATION", authorization), mock.patch.object(RUNNER, "CANONICAL_MUTAGEN_GATE", gate), mock.patch.object(RUNNER, "OUTPUT_SUBDIR", output_subdir):
                RUNNER.freeze(ROOT, registration); RUNNER.authorize(ROOT, registration, authorization)
                gate.write_text(json.dumps({"schema": RUNNER.MUTAGEN_GATE_SCHEMA, "status": "PASS_CONNECTED_FLUSHED", "registration": RUNNER.BASE.bind(registration), "authorization": RUNNER.BASE.bind(authorization), "label": RUNNER.LABEL, "gate_generator": RUNNER.BASE.bind(Path(RUNNER.__file__).resolve()), "session_name": RUNNER.MUTAGEN_NAME, "session_identifier": RUNNER.MUTAGEN_IDENTIFIER, "alpha_connected": True, "beta_connected": True, "transition_problems": 0, "flush_exit_code": 0}) + "\n")
                caps = {name: "1" for name in RUNNER.BASE.THREAD_CAPS}
                with mock.patch.dict(os.environ, caps, clear=False), mock.patch.object(RUNNER.socket, "gethostname", return_value=RUNNER.REMOTE_HOSTNAME), mock.patch.object(RUNNER.platform, "system", return_value="Darwin"), mock.patch.object(RUNNER.platform, "machine", return_value=RUNNER.REMOTE_MACHINE), mock.patch.object(RUNNER.shutil, "disk_usage", return_value=SimpleNamespace(free=9 * 1024**3)), mock.patch.object(RUNNER, "nonblocking_lock_probe", return_value=nullcontext()):
                    report = RUNNER.preflight(ROOT, registration, authorization, gate)
            self.assertEqual(report["status"], "PASS_R5B_ZERO_OPENMM_REMOTE_PREFLIGHT")
            self.assertEqual(report["new_capture_queries"], 0)


if __name__ == "__main__": unittest.main()
