"""Zero-OpenMM tests for schema-aware selected full-state replay R4."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace
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
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RECORDER = load("bb_full_state_recorder_r4_test", ROOT / "scripts/full_state_history_recorder_r4.py")
RUNNER = load("bb_full_state_runner_r4_test", ROOT / "scripts/run_selected_ivm_full_state_history_replay_r4.py")
VERIFIER = load("bb_full_state_verifier_r4_test", ROOT / "scripts/verify_selected_ivm_full_state_history_replay_r4.py")


def counter_mapping(nf: int, include_inherited: bool = True):
    joint = np.zeros(32, dtype=np.int64)
    joint[1:-1] = nf // 30
    endpoint = np.zeros(32, dtype=np.int64)
    endpoint[[0, -1]] = 1
    requests = joint + endpoint
    result = {
        "joint_evaluations": joint,
        "energy_requests": requests.copy(),
        "force_requests": requests.copy(),
        "cache_hits": endpoint.copy(),
        "total_query_events": requests.copy(),
        "both_requests": requests.copy(),
        "budget_blocked_misses": np.zeros(32, dtype=np.int64),
    }
    if include_inherited:
        result["inherited_cache_entries"] = endpoint.copy()
    return result


def physical_state(call: int):
    path = np.zeros((32, 9396), dtype=np.float64)
    path[:, 0] = np.arange(32, dtype=np.float64) + call
    energy = np.arange(32, dtype=np.float64) + call
    force = np.full((32, 9396), float(call), dtype=np.float64)
    return path, energy, force, {"i_perp": float(call)}


class FullStateHistoryReplayR4Test(unittest.TestCase):
    def setUp(self):
        selected = json.loads((ROOT / "registrations/production/bb_irivm_a1e0_r01_m05_inf9990_r1.json").read_text(encoding="utf-8"))
        self.reference = [str(value) for value in selected["command"]]

    def make_recorder(self, directory: Path):
        registration = directory / "registration.json"
        registration.write_text("{}\n", encoding="utf-8")
        return RECORDER.FullStateHistoryRecorder(
            directory / "observer", RUNNER.LABEL, registration,
            RECORDER.sha256_file(registration),
        )

    def test_runtime_delta_is_only_label_and_output(self):
        replay = RUNNER.replay_argv({"command": self.reference})
        RUNNER.validate_runtime_delta(self.reference, replay)
        self.assertEqual(
            [(left, right) for left, right in zip(self.reference, replay) if left != right],
            [
                ("runs/production", "runs/full_state_history_replay_r4"),
                ("bb_irivm_a1e0_r01_m05_inf9990_r1", "bb_irivm_a1e0_r01_m05_inf9990_fullstate_r4"),
            ],
        )
        self.assertEqual(RUNNER.PATH_ONLY.option_value(replay, "--path-storage-mode"), "committed")
        self.assertNotIn("--save-all-paths", replay)

    def test_outer0_historical_schema7_then_reparam_live_schema8(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = self.make_recorder(Path(temporary))

            def should_store_path(**kwargs):
                return bool(kwargs["is_initial"] or kwargs["reparam_committed"])

            should_store_path.__name__ = "should_store_path"
            should_store_path.__module__ = "storage_policy"
            fake_runner = None

            def run():
                nonlocal fake_runner
                path, energies, forces, diagnostic = physical_state(1)
                live_counters = counter_mapping(30, True)
                history = {"calls_per_image": [], "total_internal_joint_evaluations": []}

                def record(counter_snapshot=None, reparam_committed=False):
                    counters = live_counters if counter_snapshot is None else counter_snapshot
                    call = int(np.max(counters["joint_evaluations"][1:-1]))
                    history["calls_per_image"].append(call)
                    history["total_internal_joint_evaluations"].append(int(np.sum(counters["joint_evaluations"][1:-1])))
                    _ = path, diagnostic
                    return fake_runner.should_store_path(
                        mode="committed", save_all_paths=False, call=call,
                        is_initial=(call == 1), reparam_committed=reparam_committed,
                    )

                def attempt_reparameterization():
                    nonlocal path, energies, forces, diagnostic, live_counters
                    path, energies, forces, diagnostic = physical_state(6)
                    live_counters = counter_mapping(180, True)
                    _ = path, energies, forces, diagnostic
                    return record(counter_snapshot=None, reparam_committed=True)

                record()
                # The real first frozen-discrete solve computes all four
                # states before the caller publishes these trace snapshots.
                snapshots = [counter_mapping(nf, False) for nf in (60, 90, 120, 150)]
                for call, snapshot in enumerate(snapshots, start=2):
                    path, energies, forces, diagnostic = physical_state(call)
                    record(counter_snapshot=snapshot)
                attempt_reparameterization()

            fake_runner = SimpleNamespace(
                __file__=str(Path(__file__).resolve()), run=run,
                should_store_path=should_store_path,
            )
            restore = RECORDER.install_output_observer(
                fake_runner, recorder,
                expected_source_sha256=RECORDER.sha256_file(Path(__file__).resolve()),
            )
            try:
                run()
            finally:
                restore()
            self.assertEqual([row["optimization_N_F"] for row in recorder.records], [30, 60, 90, 120, 150, 180])
            self.assertEqual([row["counter_source_kind"] for row in recorder.records], ["live_counter_arrays", *("historical_counter_snapshot" for _ in range(4)), "live_counter_arrays"])
            for index in range(1, 5):
                with np.load(recorder.records[index]["shard"], allow_pickle=False) as shard:
                    self.assertFalse(bool(shard["inherited_cache_entries_present"]))
                    self.assertNotIn("inherited_cache_entry_vector", shard.files)
                    self.assertEqual(shard["unavailable_counter_fields"].astype(str).tolist(), ["inherited_cache_entries"])
                    self.assertEqual(shard["counter_schema_keys"].astype(str).tolist(), list(RECORDER.COMMON_COUNTER_KEYS))
            for index in (0, 5):
                with np.load(recorder.records[index]["shard"], allow_pickle=False) as shard:
                    self.assertTrue(bool(shard["inherited_cache_entries_present"]))
                    self.assertIn("inherited_cache_entry_vector", shard.files)
                    self.assertEqual(shard["unavailable_counter_fields"].size, 0)
                    self.assertEqual(shard["unavailable_counter_fields"].dtype.kind, "U")

    def test_explicit_historical_schema8_is_stored_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = self.make_recorder(Path(temporary))
            path, energy, force, diagnostic = physical_state(1)
            counters = counter_mapping(30, True)
            history = {"calls_per_image": [1], "total_internal_joint_evaluations": [30]}
            recorder.record(1, path, energy, force, counters, history, diagnostic, counter_source_kind="historical_counter_snapshot")
            with np.load(recorder.records[0]["shard"], allow_pickle=False) as shard:
                self.assertEqual(str(shard["counter_source_kind"]), "historical_counter_snapshot")
                self.assertTrue(bool(shard["inherited_cache_entries_present"]))
                self.assertTrue(np.array_equal(shard["inherited_cache_entry_vector"], counters["inherited_cache_entries"]))

    def test_wrong_missing_extra_and_noninteger_counter_schemas_fail_closed(self):
        variants = []
        live7 = counter_mapping(30, False)
        variants.append(("live_counter_arrays", live7, "exact schema-8"))
        missing = counter_mapping(30, False); missing.pop("both_requests")
        variants.append(("historical_counter_snapshot", missing, "neither exact schema-7"))
        extra = counter_mapping(30, False); extra["invented"] = np.zeros(32, dtype=np.int64)
        variants.append(("historical_counter_snapshot", extra, "neither exact schema-7"))
        floats = counter_mapping(30, False); floats["both_requests"] = floats["both_requests"].astype(np.float64)
        variants.append(("historical_counter_snapshot", floats, "dtype is not integer"))
        for source, counters, message in variants:
            with self.subTest(source=source, message=message), tempfile.TemporaryDirectory() as temporary:
                recorder = self.make_recorder(Path(temporary))
                path, energy, force, diagnostic = physical_state(1)
                history = {"calls_per_image": [1], "total_internal_joint_evaluations": [30]}
                with self.assertRaisesRegex(RuntimeError, message):
                    recorder.record(1, path, energy, force, counters, history, diagnostic, counter_source_kind=source)
                self.assertEqual(recorder.records, [])
                self.assertEqual(list(recorder.shards.iterdir()), [])

    def test_verifier_accepts_schema7_without_numeric_inherited_and_closes_nf(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            recorder = self.make_recorder(directory)
            path, energy, force, _ = physical_state(1)
            i_perp = VERIFIER.BASE.discrete_i_perp(path, force)
            diagnostic = {"i_perp": i_perp}
            counters = counter_mapping(30, False)
            history = {"calls_per_image": [1], "total_internal_joint_evaluations": [30]}
            recorder.record(1, path, energy, force, counters, history, diagnostic, counter_source_kind="historical_counter_snapshot")
            manifest_path = recorder.root / "full_state_history.manifest.json"
            manifest = {
                "counter_schema_policy": "exact live-8 or historical-7/8; absent inherited values remain unavailable",
                "counter_schema_counts": {"live_schema8": 0, "historical_schema7": 1, "historical_schema8": 0},
                "records": recorder.records,
            }
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
            RECORDER.append_fsync_jsonl(recorder.wal, {
                "event": "manifest_committed", "record_count": 1,
                "manifest": str(manifest_path), "manifest_sha256": VERIFIER.BASE.sha256_file(manifest_path),
            })
            capture_audit = {
                "wal": str(recorder.wal), "wal_sha256": VERIFIER.BASE.sha256_file(recorder.wal),
                "counter_schema_counts": manifest["counter_schema_counts"],
                "missing_counter_values_guessed_or_backfilled": False,
            }
            arrays = {
                "stored_calls": np.asarray([1], dtype=np.int64),
                "stored_paths": np.asarray([path]),
                "nf": np.asarray([30], dtype=np.int64),
                "i_perp": np.asarray([i_perp]),
                "barrier_kcal": np.asarray([(np.max(energy) - energy[0]) * 23.0605]),
                "counters": np.asarray([counters["joint_evaluations"]]),
                "energy_requests": np.asarray([counters["energy_requests"]]),
                "force_requests": np.asarray([counters["force_requests"]]),
                "cache_hits": np.asarray([counters["cache_hits"]]),
            }
            registration_path = (directory / "registration.json").resolve(strict=True)
            with mock.patch.object(VERIFIER, "ROWS", 1), mock.patch.object(VERIFIER, "LAST_COMPLETE_CALL", 1):
                inventory, summary = VERIFIER.verify_shards_r4(registration_path, manifest_path, capture_audit, manifest, arrays)
                self.assertEqual(summary["counter_schema_counts"]["historical_schema7"], 1)
                self.assertEqual(inventory[0]["unavailable_counter_fields"], ["inherited_cache_entries"])
                bad_arrays = dict(arrays)
                bad_arrays["nf"] = np.asarray([31], dtype=np.int64)
                with self.assertRaisesRegex(RuntimeError, "WAL N_F mismatch"):
                    VERIFIER.verify_shards_r4(registration_path, manifest_path, capture_audit, manifest, bad_arrays)

    def test_actual_production_source_has_exact_frame_code_inventory_without_import(self):
        source_path = (ROOT / "src/run_ivm_v9_spring_projected_production_20260730.py").resolve(strict=True)
        module_code = compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec")
        run_codes = [constant for constant in module_code.co_consts if isinstance(constant, CodeType) and constant.co_name == "run"]
        self.assertEqual(len(run_codes), 1)
        fake_module = SimpleNamespace(__file__=str(source_path), run=FunctionType(run_codes[0], {}))
        contract = RECORDER.freeze_frame_code_contract(fake_module)
        self.assertEqual(contract.record_code.co_name, "record")
        self.assertEqual({code.co_name for code in contract.allowed_state_codes}, {"run", "attempt_reparameterization"})
        frozen_source = (ROOT / "src/frozen_discrete_b040.py").read_text(encoding="utf-8")
        self.assertIn('"budget_blocked_misses",\n)', frozen_source)
        self.assertNotIn('"inherited_cache_entries",\n)', frozen_source.split("class DiscreteB040SafetyError", 1)[0])

    def test_wrong_immediate_caller_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = self.make_recorder(Path(temporary))

            def should_store_path(**_kwargs): return True
            should_store_path.__name__ = "should_store_path"; should_store_path.__module__ = "storage_policy"
            fake_runner = None

            def run(return_record):
                path, energies, forces, diagnostic = physical_state(1)
                live_counters = counter_mapping(30, True)
                history = {"calls_per_image": [1], "total_internal_joint_evaluations": [30]}
                def record(counter_snapshot=None):
                    counters = live_counters if counter_snapshot is None else counter_snapshot
                    _ = path, diagnostic, counters, history
                    return fake_runner.should_store_path(mode="committed", save_all_paths=False, call=1, is_initial=True, reparam_committed=False)
                def attempt_reparameterization():
                    nonlocal path, energies, forces, diagnostic
                    _ = path, energies, forces, diagnostic
                    return record()
                return record if return_record else attempt_reparameterization()

            fake_runner = SimpleNamespace(__file__=str(Path(__file__).resolve()), run=run, should_store_path=should_store_path)
            restore = RECORDER.install_output_observer(fake_runner, recorder, expected_source_sha256=RECORDER.sha256_file(Path(__file__).resolve()))
            closure = run(True)
            def wrong_caller(): return closure()
            try:
                with self.assertRaisesRegex(RuntimeError, "allowed hash-bound state frame"):
                    wrong_caller()
            finally:
                restore()
            self.assertEqual(recorder.records, [])

    def test_freeze_authorize_and_preflight_new_R4_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            registration_path = Path(temporary) / "registration.json"
            authorization_path = Path(temporary) / "authorization.json"
            RUNNER.freeze(ROOT, registration_path)
            RUNNER.authorize(ROOT, registration_path, authorization_path)
            registration = json.loads(registration_path.read_text(encoding="utf-8"))
            authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
            self.assertEqual(registration["schema"], RUNNER.SCHEMA)
            self.assertTrue(registration["R2_and_R3_reuse_forbidden"])
            self.assertEqual(registration["schema_aware_output_only_capture"]["schema7_storage"], "numeric inherited field absent; presence=false; unavailable_counter_fields records the omission")
            self.assertTrue(authorization["R2_and_R3_authorization_reuse_forbidden"])
            with mock.patch.dict(os.environ, {name: "1" for name in RUNNER.PATH_ONLY.THREAD_CAPS}, clear=False), mock.patch.object(
                RUNNER.shutil, "disk_usage", return_value=SimpleNamespace(total=20 * 1024**3, used=10 * 1024**3, free=10 * 1024**3),
            ):
                RUNNER.preflight(ROOT, registration_path, authorization_path)


if __name__ == "__main__":
    unittest.main()
