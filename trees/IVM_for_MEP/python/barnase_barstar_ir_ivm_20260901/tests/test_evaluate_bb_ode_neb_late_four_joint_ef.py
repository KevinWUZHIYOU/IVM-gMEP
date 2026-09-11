"""Zero-physical tests for the independent ODE--NEB late-four package."""

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
    "bb_ode_late_four_test_evaluator",
    ROOT / "scripts/evaluate_bb_ode_neb_late_four_joint_ef.py",
)
FREEZER = load(
    "bb_ode_late_four_test_freezer",
    ROOT / "scripts/freeze_bb_ode_neb_late_four_joint_ef_contract.py",
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


class LateFourTest(unittest.TestCase):
    def setUp(self):
        FakeBackend.constructions = 0
        FakeBackend.evaluations = 0
        FakeBackend.interrupt_at = None

    def synthetic_context(self):
        stack = ExitStack()
        for module in (EVALUATOR, EVALUATOR.ENGINE):
            stack.enter_context(mock.patch.object(module, "N_IMAGES", 4))
            stack.enter_context(mock.patch.object(module, "DIMENSION", 6))
            stack.enter_context(mock.patch.object(module, "N_DENSE", 5))
        stack.enter_context(mock.patch.object(EVALUATOR, "N_INTERNAL", 2))
        stack.enter_context(mock.patch.object(EVALUATOR, "N_ATOMS", 2))
        stack.enter_context(mock.patch.object(EVALUATOR, "MAXIMUM_ATTEMPT_STARTED", 20))
        stack.enter_context(mock.patch.object(EVALUATOR, "MINIMUM_FREE_BYTES_RESERVE", 0))
        stack.enter_context(mock.patch.object(EVALUATOR, "ESTIMATED_SHARD_BYTES", 1024))
        stack.enter_context(mock.patch.object(EVALUATOR.ENGINE, "OpenMMBackend", FakeBackend))
        return stack

    def synthetic_state(self, directory: Path):
        contract = directory / "contract.json"
        contract.write_text("{}\n", encoding="utf-8")
        topology = directory / "topology.pdb"
        topology.write_text("synthetic\n", encoding="utf-8")
        output = directory / "late_four.npz"
        rows = []
        direction = np.linspace(1.0, 2.0, 6)
        for index, nf in enumerate(EVALUATOR.NF_VALUES):
            path = np.ascontiguousarray(
                np.arange(4, dtype=np.float64)[:, None] * direction[None, :]
                + index
            )
            energies = np.asarray([index + 0.1, index + 0.2])
            forces = np.ascontiguousarray(-path[1:-1])
            receipt_path = directory / f"receipt_{nf}.npz"
            np.savez(receipt_path, marker=np.asarray(nf))
            i_perp = float(
                EVALUATOR.ENGINE.path_diagnostics(
                    path, np.zeros(4, dtype=np.float64), forces
                )["i_perp"]
            )
            rows.append(
                {
                    "row_index": index,
                    "label": f"ODE-NEB_NF{nf}",
                    "method": "ODE-NEB",
                    "host_role": "local",
                    "optimization_N_F": nf,
                    "path_identifier": {
                        "optimization_N_F": nf,
                        "paid_batch_id": index + 1,
                        "source_batch_id": index + 1,
                        "disposition": "accepted",
                    },
                    "paid_force_receipt": {
                        "path": str(receipt_path.resolve()),
                        "sha256": EVALUATOR.ENGINE.sha256_file(receipt_path),
                    },
                    "path_sha256": EVALUATOR.ENGINE.sha256_array(path),
                    "official_path32_sha256": str(index) * 64,
                    "discrete_I_perp": i_perp,
                    "paid_internal_energy_sha256": EVALUATOR.ENGINE.sha256_array(energies),
                    "paid_internal_force_sha256": EVALUATOR.ENGINE.sha256_array(forces),
                    "path": path,
                    "receipt": {
                        "energies": energies,
                        "forces": forces,
                        "coordinates": path[1:-1],
                        "batch_id": index + 1,
                    },
                    "receipt_path": receipt_path.resolve(),
                }
            )
        return {
            "contract": {},
            "contract_path": contract.resolve(),
            "contract_sha256": EVALUATOR.ENGINE.sha256_file(contract),
            "topology": topology.resolve(),
            "locks": [
                (directory / "one.lock").resolve(),
                (directory / "two.lock").resolve(),
            ],
            "rows": rows,
            "result_output": output.resolve(),
            "allow_test_output": True,
        }, output.resolve()

    def test_real_freeze_and_static_preflight_construct_no_openmm(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            contract = directory / "contract.json"
            argv = [
                "freeze",
                "--output",
                str(contract),
                "--global-lock-path",
                "/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock",
                "--peer-global-lock-path",
                "/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock",
            ]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                FREEZER.main()
            payload = json.loads(contract.read_text())
            self.assertEqual(payload["schema"], EVALUATOR.SCHEMA)
            self.assertEqual(
                [row["optimization_N_F"] for row in payload["paths"]],
                [2040, 4080, 8160, 9990],
            )
            self.assertEqual(payload["physical_call_policy"]["original_discrete_new_calls"], 0)
            self.assertEqual(payload["physical_call_policy"]["maximum_attempt_started_records"], 2000)
            with mock.patch.object(
                EVALUATOR.ENGINE.OpenMMBackend,
                "__init__",
                side_effect=AssertionError("preflight constructed OpenMM"),
            ):
                state = EVALUATOR.load_contract(contract, "local")
                state["allow_test_output"] = True
                with mock.patch.object(EVALUATOR, "MINIMUM_FREE_BYTES_RESERVE", 0), mock.patch.object(EVALUATOR, "ESTIMATED_SHARD_BYTES", 0):
                    report = EVALUATOR.preflight(
                        state, directory / "future.npz", probe_locks=False
                    )
            self.assertEqual(report["paid_discrete_receipts_revalidated"], 4)
            self.assertEqual(report["original_discrete_new_calls_planned"], 0)
            self.assertEqual(report["fresh_dense_joint_EF_attempts_planned"], 2000)
            self.assertEqual(report["early_four_artifacts_touched"], 0)
            self.assertEqual(FakeBackend.constructions, 0)

    def test_synthetic_run_makes_exactly_twenty_joint_calls_and_stores_full_force(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.synthetic_context(), mock.patch.dict(
                os.environ, THREAD_CAPS, clear=False
            ):
                state, output = self.synthetic_state(directory)
                EVALUATOR.run(state, output)
                self.assertEqual(FakeBackend.constructions, 1)
                self.assertEqual(FakeBackend.evaluations, 20)
                with np.load(output, allow_pickle=False) as saved:
                    self.assertEqual(saved["energy_eV"].shape, (4, 5))
                    self.assertEqual(int(saved["original_discrete_new_calls"]), 0)
                    self.assertEqual(int(saved["dense_joint_EF_attempts"]), 20)
                for nf in EVALUATOR.NF_VALUES:
                    shard = EVALUATOR._shard_path(output, nf)
                    with np.load(shard, allow_pickle=False) as saved:
                        self.assertEqual(saved["dense_force_eV_per_A"].shape, (5, 6))
                        self.assertEqual(saved["paid_internal_force_eV_per_A"].shape, (2, 6))
                        self.assertEqual(int(saved["original_discrete_new_calls"]), 0)
                audit = json.loads(output.with_suffix(".audit.json").read_text())
                self.assertEqual(audit["dense_joint_EF_attempt_lifetime"]["attempt_started"], 20)
                self.assertEqual(audit["dense_full_force_vectors_stored"], 20)
                self.assertEqual(audit["optimization_N_F_contribution"], 0)

    def test_interrupted_path_fails_closed_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.synthetic_context():
                state, output = self.synthetic_state(directory)
                FakeBackend.interrupt_at = 3
                with self.assertRaisesRegex(KeyboardInterrupt, "synthetic interruption"):
                    EVALUATOR.run(state, output)
                self.assertFalse(output.exists())
                self.assertEqual(FakeBackend.evaluations, 3)
                FakeBackend.interrupt_at = None
                with self.assertRaisesRegex(ValueError, "superseding registration"):
                    EVALUATOR.preflight(state, output, probe_locks=False)
                self.assertEqual(FakeBackend.evaluations, 3)


if __name__ == "__main__":
    unittest.main()
