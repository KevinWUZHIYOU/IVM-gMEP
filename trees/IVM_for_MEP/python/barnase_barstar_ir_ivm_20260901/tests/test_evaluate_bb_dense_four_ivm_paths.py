"""Synthetic zero-OpenMM preflight for replay calls 2/5/10/50."""

from contextlib import redirect_stdout
import hashlib
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


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IvmDenseFourPreflightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freezer = load("freeze_ivm_dense4_test", ROOT / "scripts" / "freeze_bb_dense_ivm_four_contract.py")
        cls.adapter = load("eval_ivm_dense4_test", ROOT / "scripts" / "evaluate_bb_dense_four_ivm_paths.py")

    def test_uncommitted_fully_evaluated_rows_pass_without_openmm(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            calls = np.asarray([2, 5, 10, 50], dtype=np.int64)
            nf = np.asarray([60, 150, 300, 1329], dtype=np.int64)
            values = np.asarray([6.8e7, 2.9e7, 8.9e5, 952.0], dtype=np.float64)
            paths = np.empty((4, 32, 9396), dtype=np.float64)
            base = np.arange(32, dtype=np.float64)[:, None]
            for index in range(4):
                paths[index] = base + float(index) * 1e-6
            result = directory / "replay.npz"
            np.savez_compressed(
                result, stored_paths=paths, stored_path_calls=calls,
                calls_per_image=calls, counter_vectors=np.asarray([[0] + [int(value // 30)] * 29 + [int(value - (value // 30) * 29), 0] for value in nf], dtype=np.int64),
                i_perp=values,
            )
            audit = directory / "replay.audit.json"
            audit.write_text(json.dumps({"consistent": True}) + "\n", encoding="utf-8")
            rows = []
            for index, call in enumerate(calls):
                rows.append({"call_per_image": int(call), "optimization_N_F": int(nf[index]), "discrete_I_perp_eV2_per_A": float(values[index]), "reparameterization_committed": False, "selected_path_sha256": hashlib.sha256(np.ascontiguousarray(paths[index]).tobytes()).hexdigest(), "terminal_tail": False, "fully_evaluated_current_path": True})
            verification = directory / "verification.json"
            verification.write_text(json.dumps({"schema": "bb-irivm-selected-legacy-replay-verification-v1", "status": "PASS_EXACT_OUTPUT_ONLY_PREFIX", "new_reporting_rows": rows, "bindings": {"replay_result": {"path": str(result), "sha256": file_sha(result)}, "replay_audit": {"path": str(audit), "sha256": file_sha(audit)}}}) + "\n", encoding="utf-8")
            contract = directory / "contract.json"
            argv = ["freeze", "--output", str(contract), "--replay-result", str(result), "--replay-audit", str(audit), "--replay-verification", str(verification), "--host-role", "remote", "--global-lock-path", str(directory / "a.lock"), "--peer-global-lock-path", str(directory / "b.lock")]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                self.freezer.main()
            caps = {name: "1" for name in self.adapter.ENGINE.THREAD_VARS}
            with mock.patch.dict(os.environ, caps, clear=False), mock.patch.object(self.adapter.ENGINE, "OpenMMBackend", side_effect=AssertionError("OpenMM instantiated")), redirect_stdout(io.StringIO()) as captured:
                self.adapter.main(["--contract", str(contract), "--host-role", "remote", "--output", str(directory / "unused.npz"), "--preflight-only"])
            payload = json.loads(captured.getvalue())
            self.assertEqual(payload["status"], "ZERO_OPENMM_PREFLIGHT_PASS")
            self.assertEqual(payload["fresh_force_attempts_planned"], 0)
            self.assertEqual(payload["fresh_energy_attempts_planned"], 2000)


if __name__ == "__main__":
    unittest.main()
