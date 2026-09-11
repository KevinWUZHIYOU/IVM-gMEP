"""Zero-physical-call tests for the R2 short BB IVM arc500 continuation."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
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
    "bb_ivm_short_resume_r2_test_evaluator",
    ROOT / "scripts/evaluate_bb_dense_ivm_short_four_joint_resume_r2.py",
)
FREEZER = load(
    "bb_ivm_short_resume_r2_test_freezer",
    ROOT / "scripts/freeze_bb_dense_ivm_short_four_joint_resume_r2_contract.py",
)

THREAD_CAPS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OPENMM_CPU_THREADS": "1",
}


class FakeBackend:
    fail_with_keyboard_interrupt_at = None
    constructions = 0
    evaluations = 0

    def __init__(self, *_args, **_kwargs):
        type(self).constructions += 1
        self.raw_joint_attempts = 0
        self.raw_joint_evaluations = 0

    def evaluate_joint(self, coordinate):
        type(self).evaluations += 1
        self.raw_joint_attempts += 1
        if type(self).fail_with_keyboard_interrupt_at == type(self).evaluations:
            raise KeyboardInterrupt("synthetic interruption")
        self.raw_joint_evaluations += 1
        coordinate = np.asarray(coordinate, dtype=np.float64)
        return float(np.sum(coordinate)), np.ascontiguousarray(-coordinate)


class ResumeR2Test(unittest.TestCase):
    def setUp(self):
        FakeBackend.fail_with_keyboard_interrupt_at = None
        FakeBackend.constructions = 0
        FakeBackend.evaluations = 0

    def freeze_real_contract(self, directory: Path) -> Path:
        destination = directory / "contract.json"
        argv = [
            "freeze",
            "--output",
            str(destination),
            "--global-lock-path",
            "/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock",
            "--peer-global-lock-path",
            "/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock",
        ]
        with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            FREEZER.main()
        return destination

    def test_real_freeze_and_preflight_import_exactly_two_R1_shards_without_OpenMM(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            contract_path = self.freeze_real_contract(directory)
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            self.assertEqual(contract["schema"], EVALUATOR.SCHEMA)
            self.assertEqual(contract["status"], "FROZEN_BEFORE_ANY_R2_OPENMM_OR_PHYSICAL_CALL")
            self.assertEqual(
                [row["R2_action"] for row in contract["paths"]],
                [
                    "import_complete_R1_shard",
                    "import_complete_R1_shard",
                    "fresh_R2_complete_path",
                    "fresh_R2_complete_path",
                ],
            )
            self.assertEqual(
                [row["optimization_N_F"] for row in contract["paths"]],
                [60, 150, 300, 1329],
            )
            self.assertEqual(contract["physical_call_policy"]["fresh_R2_accepted_successes"], 1000)
            self.assertEqual(contract["physical_call_policy"]["fresh_R2_maximum_attempt_started_records"], 1000)
            self.assertEqual(contract["physical_call_policy"]["quarantined_R1_values_accepted"], 0)
            self.assertEqual(contract["optimization_accounting"]["optimization_N_F_contribution"], 0)

            with mock.patch.object(
                EVALUATOR.ENGINE.OpenMMBackend,
                "__init__",
                side_effect=AssertionError("zero-call preflight constructed OpenMM"),
            ):
                state = EVALUATOR.load_contract(contract_path.resolve(), "local")
                report = EVALUATOR.preflight(
                    state, EVALUATOR.CANONICAL_RESULT, probe_locks=False
                )
            self.assertEqual(report["imported_R1_complete_shards"], [2, 5])
            self.assertEqual(report["fresh_R2_joint_EF_attempts_planned"], 1000)
            self.assertEqual(report["fresh_R2_full_force_vectors_planned"], 1000)
            self.assertEqual(report["quarantined_R1_successes"], 35)
            self.assertEqual(report["quarantined_R1_open_attempts"], 1)
            self.assertEqual(report["quarantined_R1_values_reused"], 0)
            self.assertEqual(report["optimization_N_F_contribution"], 0)
            self.assertEqual(FakeBackend.constructions, 0)
            self.assertFalse(EVALUATOR.CANONICAL_RESULT.exists())

    def synthetic_context(self, directory: Path):
        stack = ExitStack()
        for module in (EVALUATOR, EVALUATOR.ENGINE):
            stack.enter_context(mock.patch.object(module, "N_IMAGES", 4))
            stack.enter_context(mock.patch.object(module, "DIMENSION", 6))
            stack.enter_context(mock.patch.object(module, "N_DENSE", 5))
        stack.enter_context(mock.patch.object(EVALUATOR, "FRESH_ACCEPTED_SUCCESSES", 10))
        stack.enter_context(mock.patch.object(EVALUATOR, "MAXIMUM_R2_ATTEMPT_STARTED", 10))
        stack.enter_context(mock.patch.object(EVALUATOR, "MINIMUM_FREE_BYTES_RESERVE", 0))
        stack.enter_context(mock.patch.object(EVALUATOR, "ESTIMATED_SHARD_BYTES", 1024))
        stack.enter_context(mock.patch.object(EVALUATOR.ENGINE, "OpenMMBackend", FakeBackend))
        return stack

    def synthetic_state(self, directory: Path) -> tuple[dict, Path]:
        contract_path = directory / "contract.json"
        contract_path.write_text("{}\n", encoding="utf-8")
        topology = directory / "topology.pdb"
        topology.write_text("synthetic\n", encoding="utf-8")
        source = directory / "short_replay.npz"
        np.savez(source, marker=np.asarray("synthetic"))
        source_binding = {
            "path": str(source.resolve()),
            "sha256": EVALUATOR.ENGINE.sha256_file(source),
        }
        output = directory / "resume_r2.npz"
        rows = []
        direction = np.linspace(1.0, 2.0, 6)
        for row_index, (call, nf) in enumerate(
            zip(EVALUATOR.CALLS, EVALUATOR.EXPECTED_NF)
        ):
            selected = np.ascontiguousarray(
                np.arange(4, dtype=np.float64)[:, None] * direction[None, :]
                + float(row_index)
            )
            row = {
                "label": f"IVM_call{call}_NF{nf}",
                "method": "IVM",
                "host_role": "local",
                "optimization_N_F": nf,
                "path_identifier": {
                    "calls_per_image": call,
                    "history_index": call - 1,
                    "fully_evaluated_current_path": True,
                    "terminal_tail": False,
                },
                "short_replay_source": source_binding,
                "path_sha256": EVALUATOR.ENGINE.sha256_array(selected),
                "online_discrete_I_perp": float(call),
                "path": selected,
                "i_perp": float(call),
                "source_path": source.resolve(),
                "row_index": row_index,
            }
            if call in EVALUATOR.IMPORTED_CALLS:
                dense, _ = EVALUATOR.ENGINE.uniform_arc500(selected)
                energy = np.sum(dense, axis=1) + 100.0 + row_index
                force = -dense
                delta = float(
                    (np.max(energy) - energy[0]) * EVALUATOR.ENGINE.EV_TO_KCAL
                )
                shard_path = directory / f"imported_call{call}.npz"
                arrays = {
                    "selected_path": selected,
                    "dense_coordinates_A": dense,
                    "dense_energy_eV": energy,
                    "dense_force_eV_per_A": force,
                    "dense_coordinate_sha256": np.asarray(
                        EVALUATOR.ENGINE.sha256_array(dense)
                    ),
                    "dense_energy_sha256": np.asarray(
                        EVALUATOR.ENGINE.sha256_array(energy)
                    ),
                    "dense_force_sha256": np.asarray(
                        EVALUATOR.ENGINE.sha256_array(force)
                    ),
                    "densified_Delta_E_kcal_per_mol": np.asarray(delta),
                }
                np.savez(shard_path, **arrays)
                row.update(
                    {
                        "R2_action": "import_complete_R1_shard",
                        "shard_path": shard_path.resolve(),
                        "shard": arrays,
                    }
                )
            else:
                row.update(
                    {
                        "R2_action": "fresh_R2_complete_path",
                        "shard_path": None,
                        "shard": None,
                    }
                )
            rows.append(row)
        state = {
            "contract": {},
            "contract_path": contract_path.resolve(),
            "contract_sha256": EVALUATOR.ENGINE.sha256_file(contract_path),
            "topology": topology.resolve(),
            "locks": [(directory / "one.lock").resolve(), (directory / "two.lock").resolve()],
            "rows": rows,
            "result_output": output.resolve(),
            "allow_test_output": True,
            "evidence": {
                "pause_receipt": contract_path.resolve(),
                "r1_dense_wal": contract_path.resolve(),
                "partial": {
                    "calls_per_image": 10,
                    "optimization_N_F": 300,
                    "successful_joint_energy_force_attempts": 35,
                    "open_interrupted_attempts": 1,
                    "open_attempt_id": "synthetic-r1-open",
                    "open_attempt_coordinate_sha256": "0" * 64,
                    "accepted_metric_values": 0,
                },
            },
        }
        return state, output.resolve()

    def test_synthetic_run_imports_two_and_physically_evaluates_only_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.synthetic_context(directory), mock.patch.dict(
                os.environ, THREAD_CAPS, clear=False
            ):
                state, output = self.synthetic_state(directory)
                EVALUATOR.run(state, output)
                self.assertEqual(FakeBackend.constructions, 1)
                self.assertEqual(FakeBackend.evaluations, 10)
                self.assertFalse(EVALUATOR._r2_shard_path(output, 2).exists())
                self.assertFalse(EVALUATOR._r2_shard_path(output, 5).exists())
                self.assertTrue(EVALUATOR._r2_shard_path(output, 10).exists())
                self.assertTrue(EVALUATOR._r2_shard_path(output, 50).exists())

                audit = json.loads(output.with_suffix(".audit.json").read_text())
                manifest = json.loads(output.with_suffix(".manifest.json").read_text())
                self.assertEqual(audit["status"], "PASS")
                self.assertEqual(audit["imported_R1_complete_shards"], [2, 5])
                self.assertEqual(audit["fresh_R2_paths_this_invocation"], [10, 50])
                self.assertEqual(audit["fresh_R2_joint_EF_attempts_lineage"], 10)
                self.assertEqual(audit["quarantined_R1_partial_tail"]["values_reused"], 0)
                self.assertEqual(audit["optimization_N_F_contribution"], 0)
                self.assertEqual(
                    [item["origin"] for item in manifest["shards"]],
                    [
                        "imported_complete_R1_shard",
                        "imported_complete_R1_shard",
                        "fresh_complete_R2_shard",
                        "fresh_complete_R2_shard",
                    ],
                )
                with np.load(output, allow_pickle=False) as saved:
                    self.assertEqual(saved["energy_eV"].shape, (4, 5))
                    self.assertEqual(
                        list(saved["shard_origin"].astype(str)),
                        [
                            "imported_complete_R1_shard",
                            "imported_complete_R1_shard",
                            "fresh_complete_R2_shard",
                            "fresh_complete_R2_shard",
                        ],
                    )
                    self.assertEqual(int(saved["fresh_R2_joint_EF_attempts"]), 10)
                    self.assertEqual(int(saved["quarantined_R1_values_reused"]), 0)

                dense_events = [
                    json.loads(line)
                    for line in output.with_suffix(".dense_joint_ef.wal.jsonl").read_text().splitlines()
                ]
                self.assertEqual(
                    sum(event.get("event") == "R1_partial_tail_quarantined" for event in dense_events),
                    1,
                )
                self.assertEqual(
                    sum(event.get("event") == "attempt_started" for event in dense_events),
                    10,
                )
                self.assertEqual(
                    sum(event.get("event") == "attempt_succeeded" for event in dense_events),
                    10,
                )
                with self.assertRaises(FileExistsError):
                    EVALUATOR.run(state, output)

    def test_interrupted_R2_path_is_never_reused_or_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.synthetic_context(directory):
                state, output = self.synthetic_state(directory)
                FakeBackend.fail_with_keyboard_interrupt_at = 3
                with self.assertRaisesRegex(KeyboardInterrupt, "synthetic interruption"):
                    EVALUATOR.run(state, output)
                self.assertFalse(output.exists())
                self.assertFalse(EVALUATOR._r2_shard_path(output, 10).exists())
                self.assertEqual(FakeBackend.evaluations, 3)
                FakeBackend.fail_with_keyboard_interrupt_at = None
                with self.assertRaisesRegex(
                    ValueError, "failed/open attempt requires a new superseding registration"
                ):
                    EVALUATOR.preflight(state, output, probe_locks=False)
                self.assertEqual(FakeBackend.evaluations, 3)
                events = [
                    json.loads(line)
                    for line in output.with_suffix(".dense_joint_ef.wal.jsonl").read_text().splitlines()
                ]
                starts = [event for event in events if event.get("event") == "attempt_started"]
                successes = [event for event in events if event.get("event") == "attempt_succeeded"]
                self.assertEqual(len(starts), 3)
                self.assertEqual(len(successes), 2)
                self.assertEqual(
                    sum(event.get("event") == "R1_partial_tail_quarantined" for event in events),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
