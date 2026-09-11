"""Zero-physical tests for the future remote IR--IVM late-six package."""

from __future__ import annotations

from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVALUATOR = load(
    "bb_ir_late_six_test_evaluator",
    ROOT / "scripts/evaluate_bb_irivm_late_six_joint_ef.py",
)
FREEZER = load(
    "bb_ir_late_six_test_freezer",
    ROOT / "scripts/freeze_bb_irivm_late_six_joint_ef_contract.py",
)


class FakeBackend:
    constructions = 0
    evaluations = 0
    interrupt_at = None

    def __init__(self, *_args, **_kwargs):
        type(self).constructions += 1
        self.raw_joint_attempts = 0
        self.raw_joint_evaluations = 0

    def evaluate_joint(self, coordinate):
        type(self).evaluations += 1
        self.raw_joint_attempts += 1
        if type(self).interrupt_at == type(self).evaluations:
            raise KeyboardInterrupt("synthetic interruption")
        self.raw_joint_evaluations += 1
        coordinate = np.asarray(coordinate, dtype=np.float64)
        return float(np.sum(coordinate)), np.ascontiguousarray(-coordinate)


THREAD_CAPS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OPENMM_CPU_THREADS": "1",
}


def mutagen_snapshot():
    return {
        "session_name": EVALUATOR.MUTAGEN_SESSION_NAME,
        "session_identifier": EVALUATOR.MUTAGEN_SESSION_IDENTIFIER,
        "flush_exit_code": 0,
        "status": "Watching",
        "alpha_connected": True,
        "beta_connected": True,
        "transition_problems": 0,
        "mutagen_executable": "/synthetic/mutagen",
        "mutagen_executable_sha256": "0" * 64,
    }


class LateSixTest(unittest.TestCase):
    def setUp(self):
        FakeBackend.constructions = 0
        FakeBackend.evaluations = 0
        FakeBackend.interrupt_at = None

    def make_future_evidence(self, directory: Path):
        history_rows, images, dimension = 12, 4, 6
        endpoint = np.asarray(
            [np.linspace(0.0, 1.0, dimension), np.linspace(4.0, 5.0, dimension)]
        )
        calls = np.arange(1, history_rows + 1, dtype=np.int64)
        nf = np.asarray([30, 60, 150, 1329, 2000, 4000, 6000, 8000, 8900, 8982, 9500, 9985], dtype=np.int64)
        paths = np.empty((history_rows, images, dimension), dtype=np.float64)
        for index in range(history_rows):
            fraction = np.linspace(0.0, 1.0, images)[:, None]
            path = (1.0 - fraction) * endpoint[0] + fraction * endpoint[1]
            path[1:-1] += (index + 1) * 1.0e-3
            paths[index] = path
        committed = np.ones(history_rows, dtype=bool)
        i_perp = np.linspace(1000.0, 1.0, history_rows)
        counters = np.tile(np.arange(images), (history_rows, 1)) + np.arange(history_rows)[:, None]
        force_requests = counters + 1
        result = directory / "all_paths.npz"
        np.savez(
            result,
            calls_per_image=calls,
            total_internal_joint_evaluations=nf,
            i_perp=i_perp,
            reparam_committed=committed,
            counter_vectors=counters,
            force_request_vectors=force_requests,
            stored_path_calls=calls,
            stored_paths=paths,
        )
        audit = directory / "all_paths.audit.json"
        audit.write_text("{}\n", encoding="utf-8")
        verification = directory / "verification.json"
        verification.write_text(
            json.dumps(
                {
                    "schema": EVALUATOR.SELECTOR.VERIFICATION_SCHEMA,
                    "status": EVALUATOR.SELECTOR.VERIFICATION_STATUS,
                    "bindings": {
                        "replay_result": EVALUATOR.bind(result),
                        "replay_audit": EVALUATOR.bind(audit),
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        indices = [4, 5, 6, 7, 8, 9]
        roles = [
            "nearest_unused_to_NF2000",
            "nearest_unused_to_NF4000",
            "nearest_unused_to_NF6000",
            "nearest_unused_to_NF8000",
            "nearest_unused_to_NF8982",
            "highest_legal_optimization_N_F_not_above_9990",
        ]
        selected = []
        for index, role in zip(indices, roles):
            selected.append(
                {
                    "history_index": index,
                    "call_per_image": int(calls[index]),
                    "optimization_N_F": int(nf[index]),
                    "selected_path_sha256": EVALUATOR.verified_array_sha256(paths[index]),
                    "reparameterization_committed": True,
                    "fully_evaluated_current_path": True,
                    "formally_reportable": True,
                    "terminal_uncommitted_charged_tail": False,
                    "selection_role": role,
                }
            )
        selection = directory / "selection.json"
        selection.write_text(
            json.dumps(
                {
                    "schema": EVALUATOR.SELECTOR.SELECTION_SCHEMA,
                    "status": EVALUATOR.SELECTOR.SELECTION_STATUS,
                    "selection_metric_values_consulted": False,
                    "terminal_partial_tail_selected": False,
                    "selected_path_count": 6,
                    "selected_paths": selected,
                    "bindings": {
                        "preregistration": EVALUATOR.bind(EVALUATOR.BLIND_PREREGISTRATION),
                        "selector": EVALUATOR.bind(EVALUATOR.SELECTOR_PATH),
                        "exact_all_paths_verification": EVALUATOR.bind(verification),
                        "all_paths_replay_result": EVALUATOR.bind(result),
                        "all_paths_replay_audit": EVALUATOR.bind(audit),
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return selection, endpoint

    def make_contract_and_state(self, directory: Path):
        package = directory / "package.json"
        package_payload = EVALUATOR.build_package_preregistration(
            sorted(EVALUATOR.CANONICAL_LOCKS, key=str)
        )
        package.write_text(json.dumps(package_payload, indent=2) + "\n", encoding="utf-8")
        selection, endpoint_coordinates = self.make_future_evidence(directory)
        endpoint = {
            "coordinates": endpoint_coordinates,
            "energies": np.asarray([10.0, 11.0]),
            "forces": np.asarray([-endpoint_coordinates[0], -endpoint_coordinates[1]]),
        }
        with mock.patch.object(EVALUATOR, "EXPECTED_HISTORY_ROWS", 12), mock.patch.object(EVALUATOR, "N_IMAGES", 4), mock.patch.object(EVALUATOR, "DIMENSION", 6), mock.patch.object(EVALUATOR, "_load_endpoint_cache", return_value=endpoint):
            contract_payload = EVALUATOR.build_final_contract(
                package, selection, mutagen_snapshot()
            )
        contract = directory / "contract.json"
        contract.write_text(json.dumps(contract_payload, indent=2) + "\n", encoding="utf-8")
        with mock.patch.object(EVALUATOR, "EXPECTED_HISTORY_ROWS", 12), mock.patch.object(EVALUATOR, "N_IMAGES", 4), mock.patch.object(EVALUATOR, "DIMENSION", 6), mock.patch.object(EVALUATOR, "_load_endpoint_cache", return_value=endpoint):
            state = EVALUATOR.load_contract(contract, "remote")
        output = directory / "late_six.npz"
        state["allow_test_output"] = True
        state["result_output"] = output.resolve()
        state["locks"] = [
            (directory / "synthetic_one.lock").resolve(),
            (directory / "synthetic_two.lock").resolve(),
        ]
        return state, output.resolve(), endpoint

    def synthetic_run_context(self, endpoint):
        stack = ExitStack()
        for module in (EVALUATOR, EVALUATOR.ENGINE):
            stack.enter_context(mock.patch.object(module, "N_IMAGES", 4))
            stack.enter_context(mock.patch.object(module, "DIMENSION", 6))
            stack.enter_context(mock.patch.object(module, "N_DENSE", 5))
        stack.enter_context(mock.patch.object(EVALUATOR, "N_INTERNAL", 2))
        stack.enter_context(mock.patch.object(EVALUATOR, "N_FRESH_PER_PATH", 3))
        stack.enter_context(mock.patch.object(EVALUATOR, "TOTAL_STORED_JOINT_EF", 30))
        stack.enter_context(mock.patch.object(EVALUATOR, "MAXIMUM_FRESH_ATTEMPT_STARTED", 18))
        stack.enter_context(mock.patch.object(EVALUATOR, "MINIMUM_FREE_BYTES_RESERVE", 0))
        stack.enter_context(mock.patch.object(EVALUATOR, "ESTIMATED_SHARD_BYTES", 1024))
        stack.enter_context(mock.patch.object(EVALUATOR.ENGINE, "OpenMMBackend", FakeBackend))
        stack.enter_context(mock.patch.object(EVALUATOR, "check_remote_host", return_value={"hostname": EVALUATOR.REMOTE_HOSTNAME, "platform": "Darwin", "machine": EVALUATOR.REMOTE_MACHINE}))
        return stack

    def test_package_preregistration_reads_no_result_and_binds_remote_gates(self):
        with mock.patch.object(
            EVALUATOR.ENGINE.OpenMMBackend,
            "__init__",
            side_effect=AssertionError("package prereg constructed OpenMM"),
        ):
            package = EVALUATOR.build_package_preregistration(
                sorted(EVALUATOR.CANONICAL_LOCKS, key=str)
            )
        self.assertEqual(package["schema"], EVALUATOR.PACKAGE_SCHEMA)
        self.assertFalse(package["result_or_selection_bound_now"])
        self.assertEqual(package["host_role"], "remote")
        self.assertEqual(package["remote_identity"]["hostname"], "Mac-mini.local")
        self.assertEqual(
            package["required_Mutagen_session"]["identifier"],
            EVALUATOR.MUTAGEN_SESSION_IDENTIFIER,
        )
        self.assertEqual(
            package["physical_call_policy"]["stored_joint_EF_values_total"],
            3000,
        )
        self.assertEqual(
            package["physical_call_policy"]["maximum_fresh_attempt_started_records"],
            2988,
        )
        self.assertEqual(FakeBackend.constructions, 0)

    def test_mutagen_flush_gate_is_explicit_and_fail_closed(self):
        completed = [
            mock.Mock(returncode=0, stdout="", stderr=""),
            mock.Mock(
                returncode=0,
                stdout=(
                    f"{EVALUATOR.MUTAGEN_SESSION_NAME}|"
                    f"{EVALUATOR.MUTAGEN_SESSION_IDENTIFIER}|Watching|true|true\n"
                ),
                stderr="",
            ),
            mock.Mock(returncode=0, stdout="Status: Watching for changes\n", stderr=""),
        ]
        with mock.patch.object(
            FREEZER.subprocess, "run", side_effect=completed
        ) as run:
            snapshot = FREEZER.flush_and_query_mutagen()
        self.assertEqual(snapshot["flush_exit_code"], 0)
        self.assertTrue(snapshot["alpha_connected"] and snapshot["beta_connected"])
        self.assertEqual(run.call_args_list[0].args[0][1:3], ["sync", "flush"])
        failed = mock.Mock(returncode=1, stdout="", stderr="not connected")
        with mock.patch.object(FREEZER.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "flush failed"):
                FREEZER.flush_and_query_mutagen()

    def test_synthetic_final_freeze_and_run_stores_6x500_analogue(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            state, output, endpoint = self.make_contract_and_state(directory)
            with self.synthetic_run_context(endpoint), mock.patch.dict(
                os.environ, THREAD_CAPS, clear=False
            ):
                EVALUATOR.run(state, output)
                self.assertEqual(FakeBackend.constructions, 1)
                self.assertEqual(FakeBackend.evaluations, 18)
                with np.load(output, allow_pickle=False) as saved:
                    self.assertEqual(saved["energy_eV"].shape, (6, 5))
                    self.assertEqual(int(saved["stored_joint_EF_values"]), 30)
                    self.assertEqual(int(saved["fresh_joint_EF_attempts"]), 18)
                    self.assertEqual(int(saved["inherited_endpoint_joint_EF_values"]), 12)
                    self.assertEqual(int(saved["full_force_vectors_stored"]), 30)
                    self.assertEqual(int(saved["original_discrete_new_calls"]), 0)
                for row in state["rows"]:
                    shard = EVALUATOR._shard_path(output, row)
                    with np.load(shard, allow_pickle=False) as saved:
                        self.assertEqual(saved["dense_force_eV_per_A"].shape, (5, 6))
                        np.testing.assert_array_equal(
                            saved["dense_energy_eV"][[0, -1]], endpoint["energies"]
                        )
                        np.testing.assert_array_equal(
                            saved["dense_force_eV_per_A"][[0, -1]], endpoint["forces"]
                        )
                        self.assertEqual(int(saved["fresh_dense_joint_EF_attempts"]), 3)
                audit = json.loads(output.with_suffix(".audit.json").read_text())
                self.assertFalse(audit["terminal_partial_tail_selected"])
                self.assertEqual(audit["fresh_joint_EF_attempt_lifetime"]["attempt_started"], 18)
                self.assertEqual(audit["optimization_N_F_contribution"], 0)

    def test_remote_host_gate_and_interruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            state, output, endpoint = self.make_contract_and_state(directory)
            with self.synthetic_run_context(endpoint):
                FakeBackend.interrupt_at = 2
                with self.assertRaisesRegex(KeyboardInterrupt, "synthetic interruption"):
                    EVALUATOR.run(state, output)
                self.assertFalse(output.exists())
                self.assertEqual(FakeBackend.evaluations, 2)
                FakeBackend.interrupt_at = None
                with self.assertRaisesRegex(ValueError, "superseding contract"):
                    EVALUATOR.preflight(
                        state,
                        output,
                        probe_locks=False,
                        probe_remote_host=False,
                    )
                self.assertEqual(FakeBackend.evaluations, 2)
            with mock.patch.object(EVALUATOR.socket, "gethostname", return_value="local-wrong-host"), mock.patch.object(EVALUATOR.platform, "system", return_value="Darwin"):
                with self.assertRaisesRegex(ValueError, "wrong execution host"):
                    EVALUATOR.check_remote_host(
                        {
                            "hostname": EVALUATOR.REMOTE_HOSTNAME,
                            "platform": "Darwin",
                            "machine": EVALUATOR.REMOTE_MACHINE,
                        }
                    )

    def test_complete_atomic_shard_is_reused_after_clean_between_path_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            state, output, endpoint = self.make_contract_and_state(directory)
            with self.synthetic_run_context(endpoint):
                original = EVALUATOR._evaluate_one

                def stop_before_second(state_arg, row, *args):
                    if row["row_index"] == 1:
                        raise KeyboardInterrupt("synthetic clean between-path stop")
                    return original(state_arg, row, *args)

                with mock.patch.object(
                    EVALUATOR, "_evaluate_one", side_effect=stop_before_second
                ):
                    with self.assertRaisesRegex(
                        KeyboardInterrupt, "clean between-path stop"
                    ):
                        EVALUATOR.run(state, output)
                first = EVALUATOR._shard_path(output, state["rows"][0])
                self.assertTrue(first.exists())
                self.assertEqual(FakeBackend.evaluations, 3)
                report = EVALUATOR.preflight(
                    state,
                    output,
                    probe_locks=False,
                    probe_remote_host=False,
                )
                self.assertEqual(
                    report["complete_shards_reusable"],
                    [state["rows"][0]["optimization_N_F"]],
                )
                self.assertEqual(report["fresh_joint_EF_attempts_planned"], 15)
                EVALUATOR.run(state, output)
                self.assertEqual(FakeBackend.evaluations, 18)
                audit = json.loads(output.with_suffix(".audit.json").read_text())
                self.assertEqual(
                    audit["complete_validated_shards_reused"],
                    [state["rows"][0]["optimization_N_F"]],
                )


if __name__ == "__main__":
    unittest.main()
