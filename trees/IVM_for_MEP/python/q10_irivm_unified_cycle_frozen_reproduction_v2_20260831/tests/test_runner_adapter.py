from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_ROOT))

from src.optimizer_interface import FrozenControls, OptimizerInterfaceError
from src.registered_adapter import (
    AdapterObservationError,
    RuntimeObserver,
    independent_hj_improved_tangents,
    observed_optimizer,
)
from src.run_frozen_reproduction import (
    RunnerGateError,
    _verify_environment,
    _verify_result_identity,
)


def evaluate_band_counted(label, evaluator, coordinate):
    return evaluator.query(coordinate)


def recovery_proposal_spring(optimizer, core, path, energies):
    return optimizer.hj_nearest_neighbor_spring_reparameterize(
        core, path, energies
    )


class FrozenControlsTests(unittest.TestCase):
    def test_registered_controls_and_run_ids(self) -> None:
        n10 = FrozenControls(10, 3.0e-3, "3e-3", 0.1, 7, 3000)
        n20 = FrozenControls(20, 1.0e-3, "1e-3", 0.3, 8, 3000)
        self.assertEqual(n10.run_id, "0fa57295d005c597")
        self.assertEqual(n20.run_id, "8cb8b3a582ac6f8e")
        with self.assertRaises(OptimizerInterfaceError):
            FrozenControls(10, 1.5e-3, "1.5e-3", 0.1, 7, 3000).validate()


class IndependentTangentTests(unittest.TestCase):
    def test_hj_formula_for_monotone_and_extremum_images(self) -> None:
        path = np.zeros((5, 10), dtype=np.float64)
        path[:, 0] = [0.0, 0.2, 0.8, 1.4, 2.0]
        energies = np.asarray([0.0, 2.0, 1.0, 3.0, 4.0])
        tangents = independent_hj_improved_tangents(path, energies)
        expected = np.zeros((3, 10), dtype=np.float64)
        expected[:, 0] = 1.0
        np.testing.assert_array_equal(tangents, expected)

    def test_core_proxy_fails_on_one_wrong_tangent(self) -> None:
        observer = RuntimeObserver()
        from src.registered_adapter import AuditedCoreProxy

        raw = SimpleNamespace(
            improved_tangents=lambda path, energies: np.zeros(
                (path.shape[0] - 2, path.shape[1]), dtype=np.float64
            )
        )
        proxy = AuditedCoreProxy(raw, observer)
        path = np.zeros((4, 10), dtype=np.float64)
        path[:, 0] = [0.0, 1.0, 2.0, 3.0]
        with self.assertRaises(AdapterObservationError):
            proxy.improved_tangents(path, np.arange(4, dtype=np.float64))
        self.assertEqual(observer.tangent_mismatch_count, 1)


class TransparentHookTests(unittest.TestCase):
    def test_query_and_spring_hooks_return_original_values(self) -> None:
        @dataclass
        class Ledger:
            total: int = 0
            limit: int = 10
            blocked_misses: int = 0
            evaluation_miss_trace: list = field(default_factory=list)

        class Evaluator:
            def __init__(self, ledger):
                self.ledger = ledger
                self.charged = True
                self.image_index = 1
                self.cache = {}

            def query(self, value):
                point = np.ascontiguousarray(value, dtype=np.float64)
                key = point.tobytes()
                if key in self.cache:
                    energy, gradient = self.cache[key]
                    return energy, gradient.copy()
                energy = float(np.sum(point * point))
                gradient = 2.0 * point
                self.cache[key] = (energy, gradient.copy())
                self.ledger.total += 1
                self.ledger.evaluation_miss_trace.append(
                    {
                        "sequence": self.ledger.total,
                        "article_nf": self.ledger.total,
                        "outer": -1,
                        "phase": "unassigned",
                        "image_index": 1,
                        "coordinate_sha256": hashlib.sha256(key).hexdigest(),
                    }
                )
                return energy, gradient.copy()

        class RawCore:
            @staticmethod
            def improved_tangents(path, energies):
                return independent_hj_improved_tangents(path, energies)

        raw_core = RawCore()
        alg = SimpleNamespace(
            Evaluator=Evaluator,
            _load_backend=lambda n: (raw_core, SimpleNamespace()),
        )

        def spring(core, path, energies):
            tangents = core.improved_tangents(path, energies)
            self.assertEqual(tangents.shape, path[1:-1].shape)
            return path.copy(), 0.0, 1

        def append_record(records, nf, phase, state):
            records.append((nf, phase, state))

        optimizer = SimpleNamespace(
            alg=alg,
            hj_nearest_neighbor_spring_reparameterize=spring,
            append_record=append_record,
        )
        observer = RuntimeObserver()
        ledger = Ledger()
        evaluator = Evaluator(ledger)
        coordinate = np.arange(10, dtype=np.float64)
        path = np.zeros((4, 10), dtype=np.float64)
        path[:, 0] = [0.0, 1.0, 2.0, 3.0]
        energies = np.arange(4, dtype=np.float64)
        with observed_optimizer(optimizer, observer):
            returned = evaluate_band_counted(
                "initial_band", evaluator, coordinate
            )
            spring_result = recovery_proposal_spring(
                optimizer, alg._load_backend(10)[0], path, energies
            )
        self.assertEqual(returned[0], float(np.sum(coordinate * coordinate)))
        np.testing.assert_array_equal(returned[1], 2.0 * coordinate)
        np.testing.assert_array_equal(spring_result[0], path)
        self.assertEqual(len(observer.evaluations), 1)
        self.assertEqual(observer.evaluations[0].phase, "initial_band")
        self.assertEqual(len(observer.reparameterizations), 1)
        self.assertEqual(
            observer.reparameterizations[0].caller,
            "recovery_proposal_spring",
        )
        self.assertEqual(observer.reparameterizations[0].tangent_call_count, 1)
        self.assertEqual(observer.tangent_mismatch_count, 0)


class RunnerGateTests(unittest.TestCase):
    def test_environment_requires_openblas_8_and_other_thread_vars_unset(self) -> None:
        identity = {
            "python_implementation": "CPython",
            "python_version": "3.8.20",
            "numpy_version": "x",
            "scipy_version": "y",
            "threadpoolctl_version": "z",
            "threadpool_info": [
                {
                    "user_api": "blas",
                    "internal_api": "openblas",
                    "prefix": "libopenblas",
                    "version": "v",
                    "num_threads": 8,
                    "threading_layer": "pthreads",
                    "architecture": "armv8",
                    "library_basename": "libopenblas.dylib",
                }
            ],
        }
        contract = {
            "jobs": 1,
            "thread_environment": {
                "OPENBLAS_NUM_THREADS": "8",
                "OMP_NUM_THREADS": None,
                "MKL_NUM_THREADS": None,
                "VECLIB_MAXIMUM_THREADS": None,
                "NUMEXPR_NUM_THREADS": None,
            },
            "preflight_runtime_identity": identity,
            "post_optimizer_runtime_identity": identity,
        }
        clean = {
            "PYTHONHASHSEED": "0",
            "OPENBLAS_NUM_THREADS": "8",
        }
        with mock.patch.dict(os.environ, clean, clear=True):
            with mock.patch(
                "src.run_frozen_reproduction.runtime_identity",
                return_value=identity,
            ):
                self.assertEqual(_verify_environment(contract), identity)
        with mock.patch.dict(os.environ, clean, clear=True):
            os.environ["OMP_NUM_THREADS"] = "8"
            with self.assertRaises(RunnerGateError):
                _verify_environment(contract)

    def test_result_identity_is_closed_and_manifest_bound(self) -> None:
        digest = "a" * 64
        keys = {
            "algorithm_source_sha256", "optimizer_sha256", "adapter_sha256",
            "runtime_sha256", "backend_sha256", "runner_sha256",
            "exploratory_source_sha256", "preformal_equivalence_sha256",
            "instrumentation_equivalence_sha256",
        }
        paths = {}
        source = {}
        for index, key in enumerate(sorted(keys)):
            relative = "src/id%d" % index
            paths[key] = "campaign:" + relative
            source[relative] = digest
        identity = {key: digest for key in keys}
        identity["q_sha256"] = (
            "372962a47504e6df28a20d8b022482a6835680765434b4385a6dba0e826974bc"
        )
        registration = {
            "result_identity": identity,
            "result_identity_paths": paths,
            "source_sha256": source,
            "dependency_sha256": {"repo:unused": "b" * 64},
        }
        self.assertEqual(_verify_result_identity(registration), identity)
        registration["result_identity"] = dict(identity)
        registration["result_identity"]["adapter_sha256"] = "c" * 64
        with self.assertRaises(RunnerGateError):
            _verify_result_identity(registration)


if __name__ == "__main__":
    unittest.main()
