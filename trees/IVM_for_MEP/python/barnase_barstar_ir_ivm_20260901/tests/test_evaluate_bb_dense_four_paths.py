"""Zero-OpenMM gate tests for the purpose-built four-path evaluator."""

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


ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DenseFourPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freezer = load_module(
            "freeze_bb_dense_ode_four_contract_test",
            ROOT / "scripts" / "freeze_bb_dense_ode_four_contract.py",
        )
        cls.evaluator = load_module(
            "evaluate_bb_dense_four_paths_test",
            ROOT / "scripts" / "evaluate_bb_dense_four_paths.py",
        )

    def freeze_contract(self, directory: Path) -> Path:
        contract = directory / "contract.json"
        arguments = [
            "freeze_bb_dense_ode_four_contract.py",
            "--output", str(contract),
            "--host-role", "local",
            "--global-lock-path", str(directory / "canonical.lock"),
            "--peer-global-lock-path", str(directory / "downloads.lock"),
        ]
        with mock.patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
            self.freezer.main()
        return contract

    def test_real_targeted_r2_four_paths_pass_without_openmm(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            contract = self.freeze_contract(directory)
            output = directory / "unused.npz"
            thread_env = {name: "1" for name in self.evaluator.THREAD_VARS}
            with mock.patch.dict(os.environ, thread_env, clear=False), mock.patch.object(
                self.evaluator,
                "OpenMMBackend",
                side_effect=AssertionError("preflight instantiated OpenMM"),
            ), redirect_stdout(io.StringIO()) as captured:
                self.evaluator.main(
                    [
                        "--contract", str(contract),
                        "--host-role", "local",
                        "--output", str(output),
                        "--preflight-only",
                    ]
                )
            payload = json.loads(captured.getvalue())
            self.assertEqual(payload["status"], "ZERO_OPENMM_PREFLIGHT_PASS")
            self.assertEqual(payload["optimization_N_F"], [60, 150, 300, 1320])
            self.assertEqual(payload["fresh_force_attempts_planned"], 120)
            self.assertEqual(payload["fresh_energy_attempts_planned"], 2000)
            self.assertFalse(output.exists())

    def test_thread_gate_fails_before_contract_or_openmm(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.json"
            with mock.patch.dict(os.environ, {name: "1" for name in self.evaluator.THREAD_VARS}, clear=False):
                os.environ["OPENMM_CPU_THREADS"] = "2"
                with self.assertRaisesRegex(RuntimeError, "six thread variables"):
                    self.evaluator.main(
                        [
                            "--contract", str(missing),
                            "--host-role", "local",
                            "--output", str(Path(temporary) / "unused.npz"),
                            "--preflight-only",
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
