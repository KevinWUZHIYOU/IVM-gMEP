from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
EVALUATOR_PATH = ROOT / "scripts/evaluate_bb_irivm_nf8982_0k_section_energy_r4.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("bb_nf8982_energy_tested", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeEnergyBackend:
    constructions = 0

    def __init__(self, _topology):
        type(self).constructions += 1
        self.raw_energy_attempts = 0
        self.raw_energy_successes = 0
        self.backend = type("Ledger", (), {"raw_joint_attempts": 0, "raw_joint_evaluations": 0})()

    def evaluate_energy(self, coordinate):
        self.raw_energy_attempts += 1
        value = float(np.sum(np.asarray(coordinate, dtype=np.float64)))
        self.raw_energy_successes += 1
        return value


class FailingBackend(FakeEnergyBackend):
    def evaluate_energy(self, coordinate):
        self.raw_energy_attempts += 1
        raise RuntimeError("synthetic physical failure")


class TestNF8982PathEnergy(unittest.TestCase):
    def test_r4_rejects_noncanonical_authorization_and_binds_authorizer(self):
        module = load_evaluator()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            contract_path = directory / "contract.json"
            authorization_path = directory / "authorization.json"
            wrong_authorization = directory / "copy.json"
            preauth_path = directory / "preauth.json"
            contract = {
                "execution_authorization_output": str(authorization_path),
                "execution_authorizer": module.file_binding(module.AUTHORIZER_PATH),
            }
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            preauth = {
                "canonical_two_lock_nonblocking_probe": {
                    "status": "PASS",
                    "locks": [str(path) for path in module.CANONICAL_LOCKS],
                },
                "thread_policy": {name: "1" for name in module.THREAD_VARS},
                "physical_calls_started": 0,
                "openmm_contexts_constructed": 0,
                "path_stage_fresh_attempt_maximum": module.PATH_INTERNAL_CALLS,
                "surface_stage_fresh_attempt_maximum": module.SURFACE_FRESH_CALLS,
                "total_fresh_attempt_maximum": module.TOTAL_FRESH_CALLS,
            }
            preauth_path.write_text(json.dumps(preauth), encoding="utf-8")
            authorization = {"authorizer": module.file_binding(module.AUTHORIZER_PATH)}
            authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
            wrong_authorization.write_text(json.dumps(authorization), encoding="utf-8")
            state = {
                "contract": contract,
                "authorization": authorization,
                "preauthorization_preflight": preauth_path,
            }
            with mock.patch.object(module, "_original_load_contract", return_value=state):
                accepted = module.load_contract(contract_path, authorization_path, "local", directory / "x.npz", module.PATH_STAGE)
                self.assertIs(accepted, state)
                with self.assertRaisesRegex(ValueError, "authorization path differs"):
                    module.load_contract(contract_path, wrong_authorization, "local", directory / "x.npz", module.PATH_STAGE)
            bad = dict(authorization)
            bad["authorizer"] = {"path": str(module.AUTHORIZER_PATH), "sha256": "0" * 64}
            state_bad = dict(state, authorization=bad)
            with mock.patch.object(module, "_original_load_contract", return_value=state_bad):
                with self.assertRaisesRegex(ValueError, "authorizer binding mismatch"):
                    module.load_contract(contract_path, authorization_path, "local", directory / "x.npz", module.PATH_STAGE)

    def test_real_coordinate_and_endpoint_inputs_are_exact_without_openmm(self):
        module = load_evaluator()
        coordinate = ROOT / "runs/diagnostics/bb_irivm_nf8982_0k_section_coordinates_59x28_20260902_r2.npz"
        data = module.validate_coordinate_manifest(
            coordinate,
            "9259354b5fce0fb57e040436e7800cd1fc67cd7c1cc71305663fd915057cc02e",
        )
        endpoint = module.load_endpoint_cache(
            ROOT / "inputs/endpoint_joint_ef_cache.npz",
            data,
            module.ENDPOINT_CACHE_SHA256,
        )
        self.assertEqual(data["n_grid"][module.ZERO_INDEX].tobytes(), np.float64(0.0).tobytes())
        self.assertTrue(np.array_equal(endpoint["coordinates"], data["original_path32"][[0, -1]].reshape(2, -1)))
        self.assertEqual(len(list(module._logical_coordinates(data, module.PATH_STAGE))), 30)

    def _synthetic(self, directory: Path):
        module = load_evaluator()
        patches = {
            "POINTS": 3,
            "NORMAL_POINTS": 3,
            "ZERO_INDEX": 1,
            "N_IMAGES": 4,
            "N_INTERNAL": 2,
            "PATH_INTERNAL_CALLS": 2,
            "SURFACE_CALLS": 9,
            "SURFACE_FRESH_CALLS": 7,
            "TOTAL_FRESH_CALLS": 9,
            "N_ATOMS": 2,
            "DIMENSION": 6,
            "MINIMUM_FREE_BYTES_RESERVE": 0,
            "ESTIMATED_FINAL_BYTES": 0,
            "CANONICAL_LOCKS": {
                (directory / "lock-a").resolve(),
                (directory / "lock-b").resolve(),
            },
        }
        for name, value in patches.items():
            setattr(module, name, value)
            setattr(module._base, name, value)
        path32 = np.arange(4 * 2 * 3, dtype=np.float64).reshape(4, 2, 3) / 10.0
        surface = np.arange(3 * 3 * 2 * 3, dtype=np.float64).reshape(3, 3, 2, 3) / 7.0
        surface[0, 1] = path32[0]
        surface[-1, 1] = path32[-1]
        cell_hashes = np.asarray(
            [[module.sha256_array(surface[i, j]) for j in range(3)] for i in range(3)]
        )
        for name in ("contract.json", "authorization.json", "preauth.json", "coordinates.npz", "policy.json", "endpoint.npz", "topology.pdb"):
            (directory / name).write_bytes(name.encode("utf-8"))
        output = directory / "surface.npz"
        path_output = directory / "path.npz"
        state = {
            "contract_path": directory / "contract.json",
            "contract_sha256": module.sha256_file(directory / "contract.json"),
            "authorization_path": directory / "authorization.json",
            "authorization_sha256": module.sha256_file(directory / "authorization.json"),
            "preauthorization_preflight": directory / "preauth.json",
            "topology": directory / "topology.pdb",
            "coordinate_manifest": directory / "coordinates.npz",
            "coordinate_manifest_sha256": module.sha256_file(directory / "coordinates.npz"),
            "data": {
                "surface_coordinates": surface,
                "cell_hashes": cell_hashes,
                "center": surface[:, 1],
                "n_grid": np.asarray([-0.1, 0.0, 0.1]),
                "original_path32": path32,
                "original_path32_sha256": module.sha256_array(path32),
            },
            "energy_call_policy": directory / "policy.json",
            "endpoint": {
                "path": directory / "endpoint.npz",
                "sha256": module.sha256_file(directory / "endpoint.npz"),
                "coordinates": path32[[0, -1]].reshape(2, -1),
                "energies": np.asarray([-5.0, 8.0]),
                "forces": np.zeros((2, 6)),
                "coordinate_sha256": module.sha256_array(path32[[0, -1]].reshape(2, -1)),
                "energy_sha256": module.sha256_array(np.asarray([-5.0, 8.0])),
                "force_sha256": module.sha256_array(np.zeros((2, 6))),
            },
            "locks": [(directory / "lock-a").resolve(), (directory / "lock-b").resolve()],
            "output": output,
            "path_output": path_output,
            "requested_stage": module.PATH_STAGE,
            "requested_output": path_output,
        }
        return module, state

    def test_original_path_stage_is_independent_durable_and_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            module, state = self._synthetic(Path(temporary))
            FakeEnergyBackend.constructions = 0
            audit = module.run(state, module.PATH_STAGE, FakeEnergyBackend)
            self.assertEqual(audit["fresh_energy_only_calls_this_invocation"], 2)
            self.assertEqual(audit["inherited_endpoint_energy_values"], 2)
            self.assertEqual(FakeEnergyBackend.constructions, 1)
            output = state["path_output"]
            with np.load(output, allow_pickle=False) as saved:
                self.assertTrue(np.array_equal(saved["original_path32_A"], state["data"]["original_path32"]))
                self.assertTrue(np.array_equal(saved["fresh_evaluated_mask"], [False, True, True, False]))
                self.assertTrue(np.array_equal(saved["inherited_endpoint_mask"], [True, False, False, True]))
                self.assertEqual(float(saved["original_path32_energy_eV"][0]), -5.0)
                self.assertEqual(float(saved["original_path32_energy_eV"][-1]), 8.0)
                self.assertTrue(np.all(np.isfinite(saved["original_path32_energy_eV"])))
                self.assertEqual(int(saved["fresh_energy_only_attempts"]), 2)
                self.assertEqual(int(saved["fresh_fixed_endpoint_energy_calls"]), 0)
                self.assertEqual(int(saved["force_calls"]), 0)
                self.assertEqual(int(saved["optimization_N_F_contribution"]), 0)
            wal_path = output.with_suffix(".energy.wal.jsonl")
            events, wal_state = module.read_wal(wal_path, module.expected_wal_header(state, module.PATH_STAGE))
            self.assertEqual(wal_state["lifetime"]["attempt_started"], 2)
            self.assertEqual(wal_state["lifetime"]["attempt_succeeded"], 2)
            self.assertEqual(len(wal_state["publications"]), 1)
            self.assertEqual(sum(event.get("event") == "attempt_started" for event in events), 2)
            module.validate_result(output, state, module.PATH_STAGE, wal_state)
            module.validate_audit(output.with_suffix(".audit.json"), state, module.PATH_STAGE, wal_state)

            # Completed rerun is read-only and cannot instantiate a backend.
            with mock.patch.object(FakeEnergyBackend, "__init__", side_effect=AssertionError("backend constructed")):
                again = module.run(state, module.PATH_STAGE, FakeEnergyBackend)
            self.assertEqual(again["status"], "PASS")

    def test_failed_attempt_poison_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            module, state = self._synthetic(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "synthetic physical failure"):
                module.run(state, module.PATH_STAGE, FailingBackend)
            wal = state["path_output"].with_suffix(".energy.wal.jsonl")
            lines = [json.loads(line) for line in wal.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([line.get("event") for line in lines[1:]], ["attempt_started", "attempt_failed"])
            with self.assertRaisesRegex(ValueError, "failed/open WAL"):
                module.preflight(state, module.PATH_STAGE)

    def test_result_or_wal_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            module, state = self._synthetic(Path(temporary))
            module.run(state, module.PATH_STAGE, FakeEnergyBackend)
            wal = state["path_output"].with_suffix(".energy.wal.jsonl")
            text = wal.read_text(encoding="utf-8").replace('"result_bytes":', '"result_bytes": 1, "old_result_bytes":')
            wal.write_text(text, encoding="utf-8")
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                module.preflight(state, module.PATH_STAGE)


if __name__ == "__main__":
    unittest.main()
