"""Zero-physical tests for the superseding IR late-six R2 package."""

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
    "bb_ir_late_six_r2_test_evaluator",
    ROOT / "scripts/evaluate_bb_irivm_late_six_joint_ef_r2.py",
)
FREEZER = load(
    "bb_ir_late_six_r2_test_freezer",
    ROOT / "scripts/freeze_bb_irivm_late_six_joint_ef_r2_contract.py",
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
        coordinate = np.ascontiguousarray(coordinate, dtype=np.float64)
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
        "session_name": EVALUATOR.R1.MUTAGEN_SESSION_NAME,
        "session_identifier": EVALUATOR.R1.MUTAGEN_SESSION_IDENTIFIER,
        "flush_exit_code": 0,
        "status": "Watching",
        "alpha_connected": True,
        "beta_connected": True,
        "transition_problems": 0,
        "mutagen_executable": "/synthetic/mutagen",
        "mutagen_executable_sha256": "0" * 64,
    }


class LateSixR2Test(unittest.TestCase):
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
        nf = np.asarray(
            [30, 60, 150, 1329, 2000, 4000, 6000, 8000, 8900, 8982, 9500, 9985],
            dtype=np.int64,
        )
        paths = np.empty((history_rows, images, dimension), dtype=np.float64)
        i_perp = np.empty(history_rows, dtype=np.float64)
        for index in range(history_rows):
            fraction = np.linspace(0.0, 1.0, images)[:, None]
            path = (1.0 - fraction) * endpoint[0] + fraction * endpoint[1]
            path[1:-1] += (index + 1) * 1.0e-3
            paths[index] = path
            i_perp[index] = EVALUATOR.ENGINE.path_diagnostics(
                path, np.zeros(images), -path[1:-1]
            )["i_perp"]
        counters = (
            np.tile(np.arange(images), (history_rows, 1))
            + np.arange(history_rows)[:, None]
        )
        force_requests = counters + 1
        result = directory / "all_paths.npz"
        np.savez(
            result,
            calls_per_image=calls,
            total_internal_joint_evaluations=nf,
            i_perp=i_perp,
            reparam_committed=np.ones(history_rows, dtype=bool),
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
        indices = [4, 5, 6, 7, 9, 11]
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
                    "selected_path_sha256": EVALUATOR.R1.verified_array_sha256(
                        paths[index]
                    ),
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
                        "preregistration": EVALUATOR.bind(
                            EVALUATOR.R1.BLIND_PREREGISTRATION
                        ),
                        "selector": EVALUATOR.bind(EVALUATOR.R1.SELECTOR_PATH),
                        "exact_all_paths_verification": EVALUATOR.bind(
                            verification
                        ),
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

    def shape_context(self, endpoint=None):
        stack = ExitStack()
        for module in (EVALUATOR, EVALUATOR.ENGINE):
            stack.enter_context(mock.patch.object(module, "N_IMAGES", 4))
            stack.enter_context(mock.patch.object(module, "DIMENSION", 6))
            stack.enter_context(mock.patch.object(module, "N_DENSE", 5))
        stack.enter_context(mock.patch.object(EVALUATOR, "N_INTERNAL", 2))
        stack.enter_context(
            mock.patch.object(EVALUATOR, "N_DENSE_FRESH_CANDIDATES", 3)
        )
        stack.enter_context(mock.patch.object(EVALUATOR, "MAXIMUM_FRESH_PER_PATH", 5))
        stack.enter_context(
            mock.patch.object(EVALUATOR, "MAXIMUM_FRESH_ATTEMPT_STARTED", 30)
        )
        stack.enter_context(
            mock.patch.object(
                EVALUATOR, "TOTAL_ORIGINAL_INTERNAL_STORED_JOINT_EF", 12
            )
        )
        stack.enter_context(
            mock.patch.object(EVALUATOR, "TOTAL_DENSE_STORED_JOINT_EF", 30)
        )
        stack.enter_context(mock.patch.object(EVALUATOR.R1, "EXPECTED_HISTORY_ROWS", 12))
        stack.enter_context(mock.patch.object(EVALUATOR.R1, "N_IMAGES", 4))
        stack.enter_context(mock.patch.object(EVALUATOR.R1, "DIMENSION", 6))
        if endpoint is not None:
            stack.enter_context(
                mock.patch.object(
                    EVALUATOR.R1, "_load_endpoint_cache", return_value=endpoint
                )
            )
        return stack

    def make_contract_and_state(self, directory: Path):
        selection, endpoint_coordinates = self.make_future_evidence(directory)
        endpoint = {
            "coordinates": endpoint_coordinates,
            "energies": np.asarray(
                [np.sum(endpoint_coordinates[0]), np.sum(endpoint_coordinates[1])]
            ),
            "forces": np.asarray(
                [-endpoint_coordinates[0], -endpoint_coordinates[1]]
            ),
        }
        package = directory / "package.json"
        with self.shape_context(endpoint):
            package_payload = EVALUATOR.build_package_preregistration(
                sorted(EVALUATOR.R1.CANONICAL_LOCKS, key=str)
            )
            package.write_text(json.dumps(package_payload, indent=2) + "\n")
            contract_payload = EVALUATOR.build_final_contract(
                package, selection, mutagen_snapshot()
            )
        contract = directory / "contract.json"
        contract.write_text(json.dumps(contract_payload, indent=2) + "\n")
        with self.shape_context(endpoint):
            state = EVALUATOR.load_contract(contract, "remote")
        output = (directory / "late_six_r2.npz").resolve()
        state["allow_test_output"] = True
        state["result_output"] = output
        state["locks"] = [
            (directory / "synthetic_one.lock").resolve(),
            (directory / "synthetic_two.lock").resolve(),
        ]
        return state, output, endpoint

    def synthetic_run_context(self, endpoint):
        stack = self.shape_context(endpoint)
        stack.enter_context(mock.patch.object(EVALUATOR, "MINIMUM_FREE_BYTES_RESERVE", 0))
        stack.enter_context(mock.patch.object(EVALUATOR, "ESTIMATED_SHARD_BYTES", 1024))
        stack.enter_context(
            mock.patch.object(EVALUATOR.ENGINE, "OpenMMBackend", FakeBackend)
        )
        stack.enter_context(
            mock.patch.object(
                EVALUATOR.R1,
                "check_remote_host",
                return_value={
                    "hostname": EVALUATOR.R1.REMOTE_HOSTNAME,
                    "platform": "Darwin",
                    "machine": EVALUATOR.R1.REMOTE_MACHINE,
                },
            )
        )
        return stack

    def test_package_preregistration_binds_r1_and_reads_no_future_result(self):
        with mock.patch.object(
            EVALUATOR.ENGINE.OpenMMBackend,
            "__init__",
            side_effect=AssertionError("R2 package constructed OpenMM"),
        ):
            package = EVALUATOR.build_package_preregistration(
                sorted(EVALUATOR.R1.CANONICAL_LOCKS, key=str)
            )
        self.assertEqual(package["schema"], EVALUATOR.PACKAGE_SCHEMA)
        self.assertFalse(package["result_or_selection_bound_now"])
        self.assertTrue(package["supersedes"]["R1_package"])
        self.assertFalse(package["supersedes"]["R1_physical_results_reused"])
        self.assertEqual(
            package["physical_call_policy"]["maximum_fresh_joint_EF_per_path"],
            528,
        )
        self.assertEqual(
            package["physical_call_policy"]["maximum_fresh_attempt_started_records"],
            3168,
        )
        self.assertEqual(FakeBackend.constructions, 0)

    def test_exact_overlap_is_reused_only_with_byte_identical_coordinates(self):
        path = np.asarray(
            [np.zeros(6), 0.25 * np.ones(6), 0.50 * np.ones(6), np.ones(6)]
        )
        with self.shape_context():
            plan = EVALUATOR.exact_overlap_plan(path)
        self.assertEqual(plan["original_internal_image_to_dense_point"], [1, 2])
        self.assertEqual(plan["exact_overlap_count"], 2)
        self.assertEqual(plan["fresh_joint_EF_calls"], 3)
        changed = path.copy()
        changed[1, 0] += 1.0e-5
        with self.shape_context():
            changed_plan = EVALUATOR.exact_overlap_plan(changed)
        self.assertNotEqual(
            changed_plan["original_internal_image_to_dense_point"][0], 1
        )

    def test_mutagen_flush_gate_is_explicit_and_fail_closed(self):
        completed = [
            mock.Mock(returncode=0, stdout="", stderr=""),
            mock.Mock(
                returncode=0,
                stdout=(
                    f"{EVALUATOR.R1.MUTAGEN_SESSION_NAME}|"
                    f"{EVALUATOR.R1.MUTAGEN_SESSION_IDENTIFIER}|"
                    "Watching|true|true\n"
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
        with mock.patch.object(
            FREEZER.subprocess,
            "run",
            return_value=mock.Mock(
                returncode=1, stdout="", stderr="not connected"
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "flush failed"):
                FREEZER.flush_and_query_mutagen()

    def test_synthetic_run_stores_original30_and_arc500_analogues(self):
        with tempfile.TemporaryDirectory() as temporary:
            state, output, endpoint = self.make_contract_and_state(Path(temporary))
            with self.synthetic_run_context(endpoint), mock.patch.dict(
                os.environ, THREAD_CAPS, clear=False
            ):
                EVALUATOR.run(state, output)
            self.assertEqual(FakeBackend.constructions, 1)
            self.assertEqual(FakeBackend.evaluations, 30)
            with np.load(output, allow_pickle=False) as saved:
                self.assertEqual(saved["energy_eV"].shape, (6, 5))
                self.assertEqual(int(saved["fresh_joint_EF_attempts"]), 30)
                self.assertEqual(
                    int(saved["original_internal_joint_EF_values_stored"]), 12
                )
                self.assertEqual(int(saved["dense_joint_EF_values_stored"]), 30)
            for row in state["rows"]:
                shard = EVALUATOR._shard_path(output, row)
                with np.load(shard, allow_pickle=False) as saved:
                    self.assertEqual(
                        saved["original_internal_force_eV_per_A"].shape, (2, 6)
                    )
                    self.assertEqual(saved["dense_force_eV_per_A"].shape, (5, 6))
                    recomputed = EVALUATOR.ENGINE.path_diagnostics(
                        saved["selected_path"],
                        np.zeros(4),
                        saved["original_internal_force_eV_per_A"],
                    )["i_perp"]
                    self.assertTrue(
                        EVALUATOR.same_float(recomputed, saved["discrete_I_perp"])
                    )
                    np.testing.assert_array_equal(
                        saved["dense_energy_eV"][[0, -1]], endpoint["energies"]
                    )
                    np.testing.assert_array_equal(
                        saved["dense_force_eV_per_A"][[0, -1]], endpoint["forces"]
                    )
                    self.assertEqual(
                        int(saved["optimization_N_F_contribution"]), 0
                    )
            audit = json.loads(output.with_suffix(".audit.json").read_text())
            self.assertEqual(audit["physical_attempt_lifetime"]["attempt_started"], 30)
            self.assertFalse(audit["terminal_partial_tail_selected"])

    def test_open_attempt_fails_closed_and_wrong_host_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            state, output, endpoint = self.make_contract_and_state(Path(temporary))
            with self.synthetic_run_context(endpoint):
                FakeBackend.interrupt_at = 2
                with self.assertRaisesRegex(KeyboardInterrupt, "synthetic interruption"):
                    EVALUATOR.run(state, output)
                self.assertFalse(output.exists())
                FakeBackend.interrupt_at = None
                with self.assertRaisesRegex(ValueError, "superseding contract"):
                    EVALUATOR.preflight(
                        state,
                        output,
                        probe_locks=False,
                        probe_remote_host=False,
                    )
            with mock.patch.object(
                EVALUATOR.R1.socket, "gethostname", return_value="local-wrong-host"
            ), mock.patch.object(
                EVALUATOR.R1.platform, "system", return_value="Darwin"
            ):
                with self.assertRaisesRegex(ValueError, "wrong execution host"):
                    EVALUATOR.R1.check_remote_host(
                        {
                            "hostname": EVALUATOR.R1.REMOTE_HOSTNAME,
                            "platform": "Darwin",
                            "machine": EVALUATOR.R1.REMOTE_MACHINE,
                        }
                    )

    def test_complete_atomic_shard_resumes_without_recomputation(self):
        with tempfile.TemporaryDirectory() as temporary:
            state, output, endpoint = self.make_contract_and_state(Path(temporary))
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
                self.assertEqual(FakeBackend.evaluations, 5)
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
                self.assertEqual(report["fresh_joint_EF_attempts_planned"], 25)
                EVALUATOR.run(state, output)
                self.assertEqual(FakeBackend.evaluations, 30)


if __name__ == "__main__":
    unittest.main()
