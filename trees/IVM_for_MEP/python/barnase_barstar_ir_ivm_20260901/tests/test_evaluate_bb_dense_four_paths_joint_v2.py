"""Zero-OpenMM gates for the joint-E/F arc500 v2 evaluator."""

from contextlib import redirect_stdout
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DenseJointV2PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freezer = load_module(
            "freeze_bb_dense_ode_four_joint_v2_contract_test",
            ROOT / "scripts" / "freeze_bb_dense_ode_four_joint_v2_contract.py",
        )
        cls.evaluator = load_module(
            "evaluate_bb_dense_four_paths_joint_v2_test",
            ROOT / "scripts" / "evaluate_bb_dense_four_paths_joint_v2.py",
        )

    def freeze_contract(self, directory: Path) -> Path:
        contract = directory / "contract.json"
        arguments = [
            "freeze_bb_dense_ode_four_joint_v2_contract.py",
            "--output", str(contract), "--host-role", "local",
            "--global-lock-path", str(directory / "canonical.lock"),
            "--peer-global-lock-path", str(directory / "downloads.lock"),
        ]
        with mock.patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
            self.freezer.main()
        return contract

    def test_real_four_path_contract_preflight_never_constructs_openmm(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            contract = self.freeze_contract(directory)
            output = directory / "unused.npz"
            thread_env = {name: "1" for name in self.evaluator.THREAD_VARS}
            with mock.patch.dict(os.environ, thread_env, clear=False), mock.patch.object(
                self.evaluator, "OpenMMBackend",
                side_effect=AssertionError("preflight instantiated OpenMM"),
            ), redirect_stdout(io.StringIO()) as captured:
                self.evaluator.main([
                    "--contract", str(contract), "--host-role", "local",
                    "--output", str(output), "--preflight-only",
                ])
            payload = json.loads(captured.getvalue())
            self.assertEqual(payload["status"], "ZERO_OPENMM_PREFLIGHT_PASS")
            self.assertEqual(payload["optimization_N_F"], [60, 150, 300, 1320])
            self.assertEqual(payload["original_discrete_joint_EF_attempts_planned"], 120)
            self.assertEqual(payload["dense_joint_EF_attempts_planned"], 2000)
            self.assertEqual(payload["dense_full_force_vectors_planned_for_storage"], 2000)
            self.assertEqual(payload["dense_endpoint_joint_EF_attempts_planned"], 8)
            self.assertEqual(payload["optimization_N_F_contribution"], 0)
            self.assertFalse(output.exists())

    def test_thread_gate_fails_before_contract_and_openmm(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = {name: "1" for name in self.evaluator.THREAD_VARS}
            env["OPENMM_CPU_THREADS"] = "2"
            with mock.patch.dict(os.environ, env, clear=False), self.assertRaisesRegex(RuntimeError, "six thread variables"):
                self.evaluator.main([
                    "--contract", str(Path(temporary) / "missing.json"),
                    "--host-role", "local", "--output", str(Path(temporary) / "unused.npz"),
                    "--preflight-only",
                ])

    def test_existing_wal_requires_exact_contract_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            contract = directory / "contract.json"
            contract.write_text("{}\n", encoding="utf-8")
            digest = self.evaluator.sha256_file(contract)
            wal_path = directory / "test.wal.jsonl"
            wal = self.evaluator.DurableWal(wal_path, "schema-v", contract.resolve(), digest)
            wal.append({"event": "attempt_started", "point_index": 0})
            wal.close()
            resumed = self.evaluator.DurableWal(wal_path, "schema-v", contract.resolve(), digest)
            resumed.close()
            events = [json.loads(line)["event"] for line in wal_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events, ["wal_header", "initial_session_started", "attempt_started", "resume_session_started"])
            with self.assertRaisesRegex(ValueError, "WAL contract mismatch"):
                self.evaluator.DurableWal(wal_path, "schema-v", contract.resolve(), "0" * 64)

    def synthetic_rows(self):
        rows = []
        for row_index, nf in enumerate(self.evaluator.ODE_NF):
            base = np.arange(self.evaluator.N_IMAGES, dtype=np.float64)[:, None]
            direction = np.arange(1, self.evaluator.DIMENSION + 1, dtype=np.float64)[None, :]
            path = np.ascontiguousarray(0.01 * base * direction + row_index)
            rows.append({
                "label": f"ODE-NEB_NF{nf}", "optimization_N_F": nf,
                "path_identifier": {"optimization_N_F": nf, "synthetic": True},
                "path_sha256": self.evaluator.sha256_array(path),
                "official_path32_sha256": self.evaluator.official_path32_sha256(
                    path.reshape(self.evaluator.N_IMAGES, self.evaluator.N_ATOMS, 3)
                ),
                "path": path,
            })
        return rows

    def test_fake_backend_full_storage_wal_closure_and_shard_resume(self):
        evaluator = self.evaluator

        class FakeBackend:
            instances = []

            def __init__(self, *args, **kwargs):
                self.raw_joint_attempts = 0
                self.raw_joint_evaluations = 0
                self.__class__.instances.append(self)

            def evaluate_joint(self, coordinate):
                self.raw_joint_attempts += 1
                self.raw_joint_evaluations += 1
                return float(np.dot(coordinate, coordinate)), np.ascontiguousarray(-2.0 * coordinate)

        with tempfile.TemporaryDirectory() as temporary, mock.patch.multiple(
            evaluator, N_ATOMS=2, DIMENSION=6, N_DENSE=5,
            MINIMUM_FREE_BYTES_RESERVE=0, ESTIMATED_SHARD_BYTES=1,
        ):
            directory = Path(temporary)
            contract = directory / "contract.json"
            contract.write_text("{}\n", encoding="utf-8")
            topology = directory / "topology.pdb"
            topology.write_text("synthetic\n", encoding="utf-8")
            output = directory / "result.npz"
            rows = self.synthetic_rows()
            locks = (directory / "a.lock", directory / "b.lock")
            with mock.patch.object(evaluator, "OpenMMBackend", FakeBackend):
                evaluator.run(contract.resolve(), topology.resolve(), locks, rows, output)
            self.assertEqual(sum(instance.raw_joint_attempts for instance in FakeBackend.instances), 4 * (30 + 5))
            for row in rows:
                shard_path = evaluator.shard_path_for(output, int(row["optimization_N_F"]))
                with np.load(shard_path, allow_pickle=False) as shard:
                    self.assertEqual(shard["dense_coordinates_A"].shape, (5, 6))
                    self.assertEqual(shard["dense_energy_eV"].shape, (5,))
                    self.assertEqual(shard["dense_force_eV_per_A"].shape, (5, 6))
                    self.assertEqual(evaluator.sha256_array(shard["dense_force_eV_per_A"]), str(shard["dense_force_sha256"].item()))
            audit = json.loads(output.with_suffix(".audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["WAL_shard_pointwise_closure"]["status"], "PASS")
            self.assertEqual(audit["WAL_shard_pointwise_closure"]["dense_lifetime"]["attempt_succeeded"], 20)
            self.assertEqual(audit["WAL_shard_pointwise_closure"]["dense_lifetime"]["attempt_open"], 0)
            with np.load(output, allow_pickle=False) as saved:
                tampered_arrays = {key: np.asarray(saved[key]).copy() for key in saved.files}
            tampered_arrays["energy_eV"][0, 0] += 1.0
            tampered = directory / "tampered.npz"
            np.savez(tampered, **tampered_arrays)
            shard_data = [
                evaluator._validate_shard(
                    evaluator.shard_path_for(output, int(row["optimization_N_F"])), row,
                    evaluator.sha256_file(contract), evaluator.sha256_file(topology),
                    evaluator.sha256_file(Path(evaluator.__file__)), evaluator.sha256_file(ROOT / "src" / "bb_core.py"),
                )
                for row in rows
            ]
            with self.assertRaisesRegex(ValueError, "energy_eV value mismatch"):
                evaluator._validate_aggregate(tampered, rows, shard_data, contract)
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                evaluator.run(contract.resolve(), topology.resolve(), locks, rows, output)

            output.with_suffix(".manifest.json").unlink()
            output.with_suffix(".audit.json").unlink()
            with mock.patch.object(evaluator, "OpenMMBackend", side_effect=AssertionError("validated shards must avoid physical calls")):
                evaluator.run(contract.resolve(), topology.resolve(), locks, rows, output)
            finalized = json.loads(output.with_suffix(".audit.json").read_text(encoding="utf-8"))
            self.assertEqual(finalized["complete_validated_shards_reused"], [60, 150, 300, 1320])

            output.unlink()
            output.with_suffix(".manifest.json").unlink()
            output.with_suffix(".audit.json").unlink()
            with mock.patch.object(evaluator, "OpenMMBackend", side_effect=AssertionError("validated shards must avoid physical calls")):
                evaluator.run(contract.resolve(), topology.resolve(), locks, rows, output)
            resumed = json.loads(output.with_suffix(".audit.json").read_text(encoding="utf-8"))
            self.assertEqual(resumed["complete_validated_shards_reused"], [60, 150, 300, 1320])
            self.assertEqual(resumed["individual_incomplete_WAL_attempts_reused"], 0)

    def test_failed_attempt_restarts_path_from_zero_and_is_audited(self):
        evaluator = self.evaluator

        class FailingBackend:
            def __init__(self, *args, **kwargs):
                self.raw_joint_attempts = 0
                self.raw_joint_evaluations = 0

            def evaluate_joint(self, coordinate):
                self.raw_joint_attempts += 1
                if self.raw_joint_attempts == 3:
                    raise RuntimeError("synthetic physical failure")
                self.raw_joint_evaluations += 1
                return float(np.sum(coordinate)), np.ascontiguousarray(-coordinate)

        class GoodBackend:
            def __init__(self, *args, **kwargs):
                self.raw_joint_attempts = 0
                self.raw_joint_evaluations = 0

            def evaluate_joint(self, coordinate):
                self.raw_joint_attempts += 1
                self.raw_joint_evaluations += 1
                return float(np.sum(coordinate)), np.ascontiguousarray(-coordinate)

        with tempfile.TemporaryDirectory() as temporary, mock.patch.multiple(
            evaluator, N_ATOMS=2, DIMENSION=6, N_DENSE=5,
            MINIMUM_FREE_BYTES_RESERVE=0, ESTIMATED_SHARD_BYTES=1,
        ):
            directory = Path(temporary)
            contract = directory / "contract.json"
            contract.write_text("{}\n", encoding="utf-8")
            topology = directory / "topology.pdb"
            topology.write_text("synthetic\n", encoding="utf-8")
            output = directory / "failed_then_resumed.npz"
            rows = self.synthetic_rows()
            locks = (directory / "a.lock", directory / "b.lock")
            with mock.patch.object(evaluator, "OpenMMBackend", FailingBackend), self.assertRaisesRegex(RuntimeError, "synthetic physical failure"):
                evaluator.run(contract.resolve(), topology.resolve(), locks, rows, output)
            self.assertFalse(evaluator.shard_path_for(output, 60).exists())
            with mock.patch.object(evaluator, "OpenMMBackend", GoodBackend):
                evaluator.run(contract.resolve(), topology.resolve(), locks, rows, output)
            audit = json.loads(output.with_suffix(".audit.json").read_text(encoding="utf-8"))
            lifetime = audit["WAL_shard_pointwise_closure"]["discrete_lifetime"]
            self.assertEqual(lifetime["attempt_failed"], 1)
            self.assertEqual(lifetime["attempt_open"], 0)
            self.assertEqual(lifetime["retry_attempts"], 3)
            self.assertEqual(audit["individual_incomplete_WAL_attempts_reused"], 0)


if __name__ == "__main__":
    unittest.main()
