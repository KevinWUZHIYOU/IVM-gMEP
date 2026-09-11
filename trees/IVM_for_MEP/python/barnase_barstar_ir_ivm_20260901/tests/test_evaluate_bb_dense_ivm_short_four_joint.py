"""Independent zero-OpenMM tests for the short-prefix IVM joint-E/F chain."""

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
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVALUATOR = load(
    "bb_ivm_short_four_joint_test_evaluator",
    ROOT / "scripts/evaluate_bb_dense_ivm_short_four_joint.py",
)
FREEZER = load(
    "bb_ivm_short_four_joint_test_freezer",
    ROOT / "scripts/freeze_bb_dense_ivm_short_four_joint_contract.py",
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
    fail_at = None

    def __init__(self, *_args, **_kwargs):
        self.raw_joint_attempts = 0
        self.raw_joint_evaluations = 0

    def evaluate_joint(self, coordinate):
        self.raw_joint_attempts += 1
        if self.fail_at == self.raw_joint_attempts:
            raise RuntimeError("synthetic backend failure")
        self.raw_joint_evaluations += 1
        coordinate = np.asarray(coordinate, dtype=np.float64)
        return float(np.sum(coordinate)), np.ascontiguousarray(-coordinate)


class ShortFourJointTest(unittest.TestCase):
    def freeze_real_contract(self, directory: Path) -> Path:
        contract = directory / "contract.json"
        arguments = [
            "freeze",
            "--output",
            str(contract),
            "--host-role",
            "local",
            "--global-lock-path",
            "/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock",
            "--peer-global-lock-path",
            "/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock",
        ]
        with mock.patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
            FREEZER.main()
        return contract

    def test_real_freeze_to_preflight_has_no_r4_or_openmm_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            contract_path = self.freeze_real_contract(directory)
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            self.assertEqual(contract["schema"], EVALUATOR.SCHEMA)
            self.assertEqual(
                [row["path_identifier"]["calls_per_image"] for row in contract["paths"]],
                [2, 5, 10, 50],
            )
            self.assertEqual(
                [row["optimization_N_F"] for row in contract["paths"]],
                [60, 150, 300, 1329],
            )
            self.assertEqual(
                contract["host_local_lock_paths"],
                [
                    "/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock",
                    "/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock",
                ],
            )
            physical = contract["physical_call_policy"]
            self.assertEqual(physical["original_discrete_new_calls_per_path"], 0)
            self.assertEqual(physical["dense_joint_EF_per_path"], 500)
            self.assertTrue(physical["dense_full_force_storage"])
            self.assertEqual(physical["optimization_N_F_contribution"], 0)
            durability = contract["durability_policy"]
            self.assertTrue(durability["atomic_complete_path_shard"])
            self.assertTrue(durability["shard_contains_dense_coordinates_energy_full_force"])
            self.assertFalse(durability["reuse_individual_incomplete_WAL_successes"])

            output = directory / "unused.npz"
            argv = [
                "evaluate",
                "--contract",
                str(contract_path),
                "--host-role",
                "local",
                "--output",
                str(output),
                "--preflight-only",
            ]
            stream = io.StringIO()
            # Keep the backend class-level immutable topology identity visible
            # to ``load_contract`` while making any attempted construction fail.
            with mock.patch.dict(os.environ, THREAD_CAPS, clear=False), mock.patch.object(
                EVALUATOR.ENGINE.OpenMMBackend,
                "__init__",
                side_effect=AssertionError("preflight constructed OpenMM backend"),
            ), mock.patch.object(sys, "argv", argv), redirect_stdout(stream):
                EVALUATOR.main()
            summary = json.loads(stream.getvalue())
            self.assertEqual(summary["status"], "ZERO_OPENMM_PREFLIGHT_PASS")
            self.assertEqual(summary["dense_joint_EF_attempts_planned"], 2000)
            self.assertEqual(summary["dense_full_force_vectors_planned_for_storage"], 2000)
            self.assertEqual(summary["original_discrete_new_calls_planned"], 0)
            self.assertFalse(output.exists())

            sources = (
                (ROOT / "scripts/evaluate_bb_dense_ivm_short_four_joint.py").read_text()
                + (ROOT / "scripts/freeze_bb_dense_ivm_short_four_joint_contract.py").read_text()
            )
            for forbidden in (
                "fullstate_r4",
                "full_state_history_replay_r4",
                "selected-full-state-history-verification-v4",
                "bb-irivm-full-state-history-recorder-v2",
            ):
                self.assertNotIn(forbidden, sources)

    def synthetic_rows(self, directory: Path):
        source = directory / "short_result.npz"
        np.savez(source, marker=np.asarray("synthetic"))
        source_binding = {
            "path": str(source.resolve()),
            "sha256": EVALUATOR.ENGINE.sha256_file(source),
        }
        rows = []
        direction = np.linspace(1.0, 2.0, 6)
        for row_index, (call, nf) in enumerate(
            zip(EVALUATOR.CALLS, EVALUATOR.EXPECTED_NF)
        ):
            path = np.ascontiguousarray(
                np.arange(4, dtype=np.float64)[:, None] * direction[None, :]
                + float(row_index)
            )
            rows.append(
                {
                    "label": f"IVM_call{call}_NF{nf}",
                    "optimization_N_F": nf,
                    "path_identifier": {
                        "calls_per_image": call,
                        "history_index": call - 1,
                        "fully_evaluated_current_path": True,
                        "terminal_tail": False,
                    },
                    "short_replay_source": source_binding,
                    "path_sha256": EVALUATOR.ENGINE.sha256_array(path),
                    "path": path,
                    "i_perp": float(call),
                    "source_path": source.resolve(),
                }
            )
        return rows

    def synthetic_context(self, directory: Path):
        stack = ExitStack()
        for module in (EVALUATOR, EVALUATOR.ENGINE):
            stack.enter_context(mock.patch.object(module, "N_IMAGES", 4))
            stack.enter_context(mock.patch.object(module, "DIMENSION", 6))
            stack.enter_context(mock.patch.object(module, "N_DENSE", 5))
        stack.enter_context(mock.patch.object(EVALUATOR, "MINIMUM_FREE_BYTES_RESERVE", 0))
        stack.enter_context(mock.patch.object(EVALUATOR, "ESTIMATED_SHARD_BYTES", 1024))
        stack.enter_context(mock.patch.object(EVALUATOR.ENGINE, "OpenMMBackend", FakeBackend))
        stack.enter_context(
            mock.patch.object(
                EVALUATOR,
                "CANONICAL_LOCKS",
                {
                    (directory / "one.lock").resolve(),
                    (directory / "two.lock").resolve(),
                },
            )
        )
        return stack

    def test_small_fake_joint_ef_full_force_wal_closure_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.synthetic_context(directory):
                contract = directory / "contract.json"
                contract.write_text("{}\n", encoding="utf-8")
                contract = contract.resolve(strict=True)
                topology = directory / "topology.pdb"
                topology.write_text("synthetic\n", encoding="utf-8")
                topology = topology.resolve(strict=True)
                locks = (directory / "one.lock", directory / "two.lock")
                rows = self.synthetic_rows(directory)
                output = directory / "ivm_short_joint.npz"
                FakeBackend.fail_at = None
                EVALUATOR.run(contract, topology, locks, rows, output)

                audit = json.loads(output.with_suffix(".audit.json").read_text())
                manifest = json.loads(output.with_suffix(".manifest.json").read_text())
                self.assertEqual(audit["status"], "PASS")
                self.assertEqual(audit["fresh_paths_this_invocation"], [2, 5, 10, 50])
                self.assertEqual(audit["complete_validated_shards_reused"], [])
                self.assertEqual(audit["dense_joint_EF_attempts_lineage"], 20)
                self.assertEqual(audit["dense_full_force_vectors_stored"], 20)
                self.assertEqual(audit["optimization_N_F_contribution"], 0)
                lifetime = audit["dense_WAL_shard_pointwise_closure"]["lifetime"]
                self.assertEqual(lifetime["attempt_started"], 20)
                self.assertEqual(lifetime["attempt_succeeded"], 20)
                self.assertEqual(lifetime["attempt_failed"], 0)
                self.assertEqual(lifetime["attempt_open"], 0)
                self.assertEqual(len(manifest["shards"]), 4)

                for call, row in zip(EVALUATOR.CALLS, rows):
                    shard_path = EVALUATOR.shard_path(output, call)
                    self.assertEqual(
                        EVALUATOR.ENGINE.sha256_file(shard_path),
                        next(
                            item["sha256"]
                            for item in manifest["shards"]
                            if item["call_per_image"] == call
                        ),
                    )
                    with np.load(shard_path, allow_pickle=False) as saved:
                        self.assertEqual(saved["dense_coordinates_A"].shape, (5, 6))
                        self.assertEqual(saved["dense_energy_eV"].shape, (5,))
                        self.assertEqual(saved["dense_force_eV_per_A"].shape, (5, 6))
                        self.assertTrue(np.all(np.isfinite(saved["dense_energy_eV"])))
                        self.assertTrue(np.all(np.isfinite(saved["dense_force_eV_per_A"])))
                        self.assertEqual(int(saved["optimization_N_F_contribution"]), 0)
                        self.assertTrue(np.array_equal(saved["selected_path"], row["path"]))

                source_events = [
                    json.loads(line)
                    for line in output.with_suffix(".source.wal.jsonl").read_text().splitlines()
                ]
                dense_events = [
                    json.loads(line)
                    for line in output.with_suffix(".dense_joint_ef.wal.jsonl").read_text().splitlines()
                ]
                self.assertEqual(
                    sum(
                        event.get("event")
                        == "verified_short_replay_path_and_Iperp_reused"
                        for event in source_events
                    ),
                    4,
                )
                self.assertEqual(
                    sum(
                        event.get("event") == "complete_shard_published"
                        for event in source_events
                    ),
                    4,
                )
                self.assertEqual(
                    sum(
                        event.get("event") == "complete_shard_published"
                        for event in dense_events
                    ),
                    4,
                )
                with self.assertRaises(FileExistsError):
                    EVALUATOR.run(contract, topology, locks, rows, output)

    def test_failure_restart_and_missing_stage_or_nonfinite_force_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.synthetic_context(directory):
                contract = directory / "contract.json"
                contract.write_text("{}\n", encoding="utf-8")
                contract = contract.resolve(strict=True)
                topology = directory / "topology.pdb"
                topology.write_text("synthetic\n", encoding="utf-8")
                topology = topology.resolve(strict=True)
                locks = (directory / "one.lock", directory / "two.lock")
                rows = self.synthetic_rows(directory)
                output = directory / "restart.npz"

                FakeBackend.fail_at = 3
                with self.assertRaisesRegex(RuntimeError, "synthetic backend failure"):
                    EVALUATOR.run(contract, topology, locks, rows, output)
                self.assertFalse(output.exists())
                self.assertFalse(EVALUATOR.shard_path(output, 2).exists())

                FakeBackend.fail_at = None
                EVALUATOR.run(contract, topology, locks, rows, output)
                audit = json.loads(output.with_suffix(".audit.json").read_text())
                lifetime = audit["dense_WAL_shard_pointwise_closure"]["lifetime"]
                self.assertGreaterEqual(lifetime["attempt_failed"], 1)
                self.assertGreaterEqual(lifetime["retry_attempts"], 1)
                self.assertEqual(lifetime["attempt_open"], 0)

                dense_wal = output.with_suffix(".dense_joint_ef.wal.jsonl")
                events = [json.loads(line) for line in dense_wal.read_text().splitlines()]
                incomplete = [
                    event
                    for event in events
                    if not (
                        event.get("event") == "path_stage_data_complete_uncommitted"
                        and int(event.get("row_index", -1)) == 0
                    )
                ]
                first_shard = EVALUATOR.shard_path(output, 2)
                with np.load(first_shard, allow_pickle=False) as saved:
                    with self.assertRaises(ValueError):
                        EVALUATOR.ENGINE._validate_stage_wal_closure(
                            incomplete,
                            0,
                            rows[0],
                            "dense_arc500_joint_EF",
                            np.asarray(saved["dense_coordinates_A"]),
                            np.asarray(saved["dense_energy_eV"]),
                            np.asarray(saved["dense_force_eV_per_A"]),
                        )

                    arrays = {key: np.asarray(saved[key]).copy() for key in saved.files}
                arrays["dense_force_eV_per_A"][0, 0] = np.nan
                arrays["dense_force_sha256"] = np.asarray(
                    EVALUATOR.ENGINE.sha256_array(arrays["dense_force_eV_per_A"])
                )
                np.savez(first_shard, **arrays)
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    EVALUATOR.validate_shard(
                        first_shard,
                        rows[0],
                        EVALUATOR.ENGINE.sha256_file(contract),
                        EVALUATOR.ENGINE.sha256_file(topology),
                    )


if __name__ == "__main__":
    unittest.main()
