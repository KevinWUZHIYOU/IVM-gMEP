"""Zero-OpenMM tests for corrected selected full-state replay R3."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from types import CodeType, FunctionType
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


RECORDER = load(
    "bb_full_state_recorder_r3_test",
    ROOT / "scripts/full_state_history_recorder_r3.py",
)
RUNNER = load(
    "bb_full_state_runner_r3_test",
    ROOT / "scripts/run_selected_ivm_full_state_history_replay_r3.py",
)


class FullStateHistoryReplayR3Test(unittest.TestCase):
    def setUp(self):
        selected = json.loads(
            (
                ROOT
                / "registrations/production/bb_irivm_a1e0_r01_m05_inf9990_r1.json"
            ).read_text(encoding="utf-8")
        )
        self.reference = [str(value) for value in selected["command"]]

    def test_runtime_delta_is_only_new_label_and_new_output(self):
        replay = RUNNER.replay_argv({"command": self.reference})
        RUNNER.validate_runtime_delta(self.reference, replay)
        self.assertEqual(
            [
                (left, right)
                for left, right in zip(self.reference, replay)
                if left != right
            ],
            [
                ("runs/production", "runs/full_state_history_replay_r3"),
                (
                    "bb_irivm_a1e0_r01_m05_inf9990_r1",
                    "bb_irivm_a1e0_r01_m05_inf9990_fullstate_r3",
                ),
            ],
        )
        self.assertEqual(
            RUNNER.PATH_ONLY.option_value(replay, "--path-storage-mode"),
            "committed",
        )
        self.assertNotIn("--save-all-paths", replay)

    def test_true_run_record_observer_two_frame_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            registration = temporary_path / "registration.json"
            registration.write_text("{}\n", encoding="utf-8")
            recorder = RECORDER.FullStateHistoryRecorder(
                temporary_path / "observer",
                RUNNER.LABEL,
                registration,
                RECORDER.sha256_file(registration),
            )

            def should_store_path(**kwargs):
                return bool(kwargs["is_initial"] or kwargs["reparam_committed"])

            should_store_path.__name__ = "should_store_path"
            should_store_path.__module__ = "storage_policy"
            fake_runner = None

            def run(call, through_attempt):
                path = np.zeros((32, 9396), dtype=np.float64)
                path[:, 0] = np.arange(32, dtype=np.float64) + call
                energies = np.arange(32, dtype=np.float64) + call
                forces = np.ones((32, 9396), dtype=np.float64) * call
                diagnostic = {"i_perp": float(call)}
                internal = np.zeros(32, dtype=np.int64)
                internal[1:-1] = call
                endpoint_hits = np.zeros(32, dtype=np.int64)
                endpoint_hits[[0, -1]] = 1
                requests = np.ones(32, dtype=np.int64) * call
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
                    "calls_per_image": list(range(1, call + 1)),
                    "total_internal_joint_evaluations": [
                        30 * value for value in range(1, call + 1)
                    ],
                }

                def record():
                    self.assertNotIn("energies", locals())
                    self.assertNotIn("forces", locals())
                    _ = path, diagnostic, counters, history
                    return fake_runner.should_store_path(
                        mode="committed",
                        save_all_paths=False,
                        call=call,
                        is_initial=(call == 1),
                        reparam_committed=(call == 2),
                    )

                def attempt_reparameterization():
                    nonlocal path, energies, forces, diagnostic
                    _ = path, energies, forces, diagnostic
                    return record()

                if through_attempt:
                    return attempt_reparameterization()
                return record()

            fake_runner = SimpleNamespace(
                __file__=str(Path(__file__).resolve()),
                run=run,
                should_store_path=should_store_path,
            )
            restore = RECORDER.install_output_observer(
                fake_runner,
                recorder,
                expected_source_sha256=RECORDER.sha256_file(Path(__file__).resolve()),
            )
            try:
                self.assertTrue(run(1, False))
                self.assertTrue(run(2, True))
            finally:
                restore()
            self.assertEqual(len(recorder.records), 2)
            for call, record_row in enumerate(recorder.records, start=1):
                with np.load(record_row["shard"], allow_pickle=False) as shard:
                    self.assertEqual(int(shard["call_per_image"]), call)
                    self.assertEqual(float(shard["energy_eV"][0]), float(call))
                    self.assertEqual(float(shard["force_eV_per_A"][0, 0]), float(call))

    def test_actual_production_source_has_exact_frame_code_inventory_without_import(self):
        source_path = (
            ROOT / "src/run_ivm_v9_spring_projected_production_20260730.py"
        ).resolve(strict=True)
        module_code = compile(
            source_path.read_text(encoding="utf-8"), str(source_path), "exec"
        )
        run_codes = [
            constant
            for constant in module_code.co_consts
            if isinstance(constant, CodeType) and constant.co_name == "run"
        ]
        self.assertEqual(len(run_codes), 1)
        fake_module = SimpleNamespace(
            __file__=str(source_path),
            run=FunctionType(run_codes[0], {}),
        )
        contract = RECORDER.freeze_frame_code_contract(fake_module)
        self.assertEqual(contract.record_code.co_name, "record")
        self.assertEqual(
            {code.co_name for code in contract.allowed_state_codes},
            {"run", "attempt_reparameterization"},
        )

    def test_wrong_immediate_state_frame_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            registration = temporary_path / "registration.json"
            registration.write_text("{}\n", encoding="utf-8")
            recorder = RECORDER.FullStateHistoryRecorder(
                temporary_path / "observer",
                RUNNER.LABEL,
                registration,
                RECORDER.sha256_file(registration),
            )

            def should_store_path(**_kwargs):
                return True

            should_store_path.__name__ = "should_store_path"
            should_store_path.__module__ = "storage_policy"
            fake_runner = None

            def run(return_record):
                path = np.zeros((32, 9396), dtype=np.float64)
                energies = np.zeros(32, dtype=np.float64)
                forces = np.zeros((32, 9396), dtype=np.float64)
                diagnostic = {"i_perp": 0.0}
                counters = {}
                history = {}

                def record():
                    _ = path, diagnostic, counters, history
                    return fake_runner.should_store_path(
                        mode="committed",
                        save_all_paths=False,
                        call=1,
                        is_initial=True,
                        reparam_committed=False,
                    )

                def attempt_reparameterization():
                    nonlocal path, energies, forces, diagnostic
                    _ = path, energies, forces, diagnostic
                    return record()

                if return_record:
                    return record
                return attempt_reparameterization()

            fake_runner = SimpleNamespace(
                __file__=str(Path(__file__).resolve()),
                run=run,
                should_store_path=should_store_path,
            )
            restore = RECORDER.install_output_observer(
                fake_runner,
                recorder,
                expected_source_sha256=RECORDER.sha256_file(Path(__file__).resolve()),
            )
            record_closure = run(True)

            def wrong_caller():
                return record_closure()

            try:
                with self.assertRaisesRegex(
                    RuntimeError, "allowed hash-bound state frame"
                ):
                    wrong_caller()
            finally:
                restore()
            self.assertEqual(recorder.records, [])
            self.assertEqual(
                len((recorder.wal).read_text(encoding="utf-8").splitlines()), 1
            )

        source = (ROOT / "scripts/full_state_history_recorder_r3.py").read_text(encoding="utf-8")
        self.assertNotIn(".query(", source)
        self.assertNotIn("getState(", source)

    def test_freeze_authorize_preflight_new_R3_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            registration_path = Path(temporary) / "registration.json"
            authorization_path = Path(temporary) / "authorization.json"
            RUNNER.freeze(ROOT, registration_path)
            RUNNER.authorize(ROOT, registration_path, authorization_path)
            registration = json.loads(registration_path.read_text(encoding="utf-8"))
            authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
            self.assertEqual(registration["status"], RUNNER.STATUS)
            self.assertTrue(registration["R2_reuse_forbidden"])
            self.assertEqual(registration["corrected_output_only_capture"]["authoritative_shard_calls_required"], list(range(1, 353)))
            self.assertEqual(registration["resource_estimate"]["minimum_free_disk_bytes_before_child"], 6 * 1024**3)
            self.assertTrue(authorization["R2_authorization_reuse_forbidden"])
            with mock.patch.dict(
                os.environ,
                {name: "1" for name in RUNNER.PATH_ONLY.THREAD_CAPS},
                clear=False,
            ), mock.patch.object(
                RUNNER.shutil,
                "disk_usage",
                return_value=SimpleNamespace(
                    total=20 * 1024**3,
                    used=10 * 1024**3,
                    free=10 * 1024**3,
                ),
            ):
                RUNNER.preflight(ROOT, registration_path, authorization_path)


if __name__ == "__main__":
    unittest.main()
