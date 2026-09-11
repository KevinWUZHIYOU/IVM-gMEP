"""Zero-OpenMM end-to-end tests for the R5B supplemental verifier."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import full_state_history_recorder_r5 as REC
import verify_selected_ivm_full_state_history_replay_r5 as PRIMARY
import verify_selected_ivm_full_state_history_replay_r5b_supplemental as SUP


def counters(call: int) -> dict:
    joint = np.zeros(32, dtype=np.int64)
    joint[1:-1] = call
    endpoint = np.zeros(32, dtype=np.int64)
    endpoint[[0, -1]] = 1
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
    path = np.zeros((32, 9396), dtype=np.float64)
    path[:, 0] = np.arange(32, dtype=np.float64)
    path[:, 2] = marker
    energy = np.arange(32, dtype=np.float64) + marker
    force = np.zeros((32, 9396), dtype=np.float64)
    force[:, 1] = marker
    return path, energy, force


class SupplementalVerifierTest(unittest.TestCase):
    def build(self, directory: Path, *, invalid_pending_schema: bool = False):
        # macOS exposes /var through /private/var; canonicalize because the
        # production verifier deliberately requires exact absolute bindings.
        directory = directory.resolve()
        registration = directory / "registration.json"
        authorization = directory / "authorization.json"
        registration.write_text("{}\n", encoding="utf-8")
        authorization.write_text("{}\n", encoding="utf-8")
        recorder = REC.FullStateHistoryRecorder(
            directory / "observer", "synthetic-r5b", registration,
            REC.sha256_file(registration),
        )
        history = {"calls_per_image": [], "total_internal_joint_evaluations": []}
        stable = {}
        stable_energy = {}
        stable_force = {}
        for call, marker in ((1, 1.0), (2, 2.0), (2, 2.5)):
            path, energy, force = state(marker)
            diagnostic = {"i_perp": PRIMARY.BASE.discrete_i_perp(path, force)}
            history["calls_per_image"].append(call)
            history["total_internal_joint_evaluations"].append(30 * call)
            recorder.record(
                call, path, energy, force, counters(call), history, diagnostic,
                counter_source_kind="live_counter_arrays",
            )
            stable[call], stable_energy[call], stable_force[call] = path, energy, force
        if invalid_pending_schema:
            with np.load(recorder.pending_path, allow_pickle=False) as saved:
                payload = {name: np.asarray(saved[name]) for name in saved.files}
            payload["counter_schema_keys"] = np.asarray(["joint_evaluations"], dtype="U64")
            with recorder.pending_path.open("wb") as stream:
                np.savez(stream, **payload)
            recorder.pending_record["pending_file_sha256_at_callback"] = REC.sha256_file(recorder.pending_path)

        calls = np.arange(1, 3, dtype=np.int64)
        nf = 30 * calls
        paths = np.asarray([stable[int(call)] for call in calls])
        counter_vectors = np.asarray([counters(int(call))["joint_evaluations"] for call in calls])
        energy_requests = np.asarray([counters(int(call))["energy_requests"] for call in calls])
        force_requests = np.asarray([counters(int(call))["force_requests"] for call in calls])
        cache_hits = np.asarray([counters(int(call))["cache_hits"] for call in calls])
        i_perp = np.asarray([PRIMARY.BASE.discrete_i_perp(stable[int(call)], stable_force[int(call)]) for call in calls])
        barriers = np.asarray([(np.max(stable_energy[int(call)]) - stable_energy[int(call)][0]) * 23.0605 for call in calls])
        replay = directory / "replay.npz"
        np.savez(
            replay, calls_per_image=calls, counter_vectors=counter_vectors,
            energy_request_vectors=energy_requests,
            force_request_vectors=force_requests, cache_hit_vectors=cache_hits,
            i_perp=i_perp, barrier_kcal=barriers,
            reparam_committed=np.zeros(2, dtype=bool),
            total_internal_joint_evaluations=nf,
            stored_path_calls=calls, stored_paths=paths,
            total_internal_nf_consumed=np.asarray(60),
            total_internal_nf_last_complete_state_index=np.asarray(1),
        )
        reference = directory / "reference.npz"
        np.savez(
            reference, calls_per_image=calls[:1],
            counter_vectors=counter_vectors[:1],
            energy_request_vectors=energy_requests[:1],
            force_request_vectors=force_requests[:1],
            cache_hit_vectors=cache_hits[:1], i_perp=i_perp[:1],
            barrier_kcal=barriers[:1], reparam_committed=np.zeros(1, dtype=bool),
            total_internal_joint_evaluations=nf[:1],
            stored_path_calls=calls[:1], stored_paths=paths[:1],
        )
        with mock.patch.object(REC, "OLD_COMPLETE_PREFIX_CALLS", 1):
            manifest, canonical, audit = recorder.finalize(replay, reference)
        replay_audit = directory / "replay.audit.json"
        replay_audit.write_text(json.dumps({"consistent": True, "result_file": str(replay)}) + "\n", encoding="utf-8")
        with mock.patch.object(PRIMARY, "PREFIX_CALLS", 1), mock.patch.object(PRIMARY, "PREFIX_NF", 30), mock.patch.object(PRIMARY.RUNNER, "validate_registration", return_value={}), mock.patch.object(PRIMARY.RUNNER, "validate_authorization", return_value={}):
            primary = PRIMARY.verify(ROOT, registration, authorization, reference, replay, replay_audit, manifest, canonical, audit)
        primary_path = directory / "primary.json"
        primary_path.write_text(json.dumps(primary, sort_keys=True) + "\n", encoding="utf-8")
        return registration, authorization, primary_path, manifest, canonical, audit

    def run_supplemental(self, bundle):
        registration, authorization, primary, manifest, canonical, audit = bundle
        with mock.patch.object(SUP.RUNNER, "validate_registration", return_value={}), mock.patch.object(SUP.RUNNER, "validate_authorization", return_value={}):
            return SUP.verify(ROOT, registration, authorization, primary, manifest, canonical, audit)

    def test_full_synthetic_closure_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = self.run_supplemental(self.build(Path(temporary)))
            self.assertEqual(report["status"], SUP.STATUS)
            self.assertEqual(report["canonical_record_count"], 2)
            self.assertEqual(report["raw_record_count"], 3)
            self.assertEqual(report["counter_schema_counts"], {"live_schema8": 2, "historical_schema7": 0, "historical_schema8": 0})

    def test_authoritative_shard_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self.build(Path(temporary))
            manifest = json.loads(bundle[3].read_text(encoding="utf-8"))
            shard_path = Path(manifest["canonical_records"][-1]["shard"])
            with np.load(shard_path, allow_pickle=False) as saved:
                payload = {name: np.asarray(saved[name]) for name in saved.files}
            payload["counter_schema_keys"] = np.asarray(["joint_evaluations"], dtype="U64")
            shard_path.chmod(0o600)
            with shard_path.open("wb") as stream:
                np.savez(stream, **payload)
            with self.assertRaisesRegex(RuntimeError, "shard binding drift"):
                self.run_supplemental(bundle)


if __name__ == "__main__":
    unittest.main()
