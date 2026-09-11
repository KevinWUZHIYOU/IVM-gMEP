#!/usr/bin/env python3
"""Zero-OpenMM tests for the BB discrete-32 geometric-action workflow."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
EVALUATOR = ROOT / "scripts/evaluate_bb_discrete32_geometric_action.py"
FREEZER = ROOT / "scripts/freeze_bb_discrete32_geometric_action_contract.py"
AUTHORIZER = ROOT / "scripts/authorize_bb_discrete32_geometric_action.py"
PLOTTER = ROOT / "scripts/plot_bb_discrete32_geometric_action.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENMM_CPU_THREADS",
):
    os.environ[variable] = "1"

action = load_module(EVALUATOR, "bb_discrete32_action_tested")


class DiscreteActionFormulaTests(unittest.TestCase):
    def test_exact_segmentwise_trapezoid(self):
        coordinates = np.asarray([[0.0, 0.0], [3.0, 4.0], [3.0, 6.0]])
        forces = np.asarray([[3.0, 4.0], [0.0, 12.0], [8.0, 15.0]])
        report = action.discrete_path_dependent_action(coordinates, forces)
        np.testing.assert_array_equal(report["segment_length_A"], [5.0, 2.0])
        np.testing.assert_array_equal(report["force_norm_eV_per_A"], [5.0, 12.0, 17.0])
        np.testing.assert_array_equal(report["action_increment_eV"], [42.5, 29.0])
        np.testing.assert_array_equal(report["arc_length_A"], [0.0, 5.0, 7.0])
        np.testing.assert_array_equal(report["cumulative_action_eV"], [0.0, 42.5, 71.5])

    def test_metric_uses_force_norm_not_energy_or_force_sign(self):
        coordinates = np.asarray([[0.0], [2.0], [3.0]])
        force = np.asarray([[1.0], [-3.0], [5.0]])
        positive = action.discrete_path_dependent_action(coordinates, force)
        negative = action.discrete_path_dependent_action(coordinates, -force)
        np.testing.assert_array_equal(positive["cumulative_action_eV"], negative["cumulative_action_eV"])
        np.testing.assert_array_equal(positive["cumulative_action_eV"], [0.0, 4.0, 8.0])

    def test_rejects_bad_shapes_nonfinite_and_degenerate_segments(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            action.discrete_path_dependent_action(np.zeros((3, 2)), np.zeros((3, 3)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            action.discrete_path_dependent_action(np.asarray([[0.0], [np.nan]]), np.ones((2, 1)))
        with self.assertRaisesRegex(ValueError, "degenerate"):
            action.discrete_path_dependent_action(np.asarray([[0.0], [0.0]]), np.ones((2, 1)))


class FrozenSourceTests(unittest.TestCase):
    def test_real_sources_are_exact_original_32_images(self):
        sources = action.load_sources()
        self.assertEqual(sources["ir"]["path"].shape, (32, action.DIMENSION))
        self.assertEqual(sources["ir"]["energy"].shape, (32,))
        self.assertNotIn("force", sources["ir"])
        self.assertEqual(sources["ode"]["path"].shape, (32, action.DIMENSION))
        self.assertEqual(sources["ode"]["force"].shape, (32, action.DIMENSION))
        self.assertEqual(action.sha256_array(sources["ir"]["path"]), action.EXPECTED_IR_PATH_SHA256)
        self.assertEqual(action.sha256_array(sources["ode"]["path"]), action.EXPECTED_ODE_PATH_SHA256)
        self.assertEqual(action.sha256_array(sources["ode"]["force"]), action.EXPECTED_ODE_FORCE_SHA256)
        np.testing.assert_array_equal(sources["ir"]["path"][[0, -1]], sources["endpoint"]["coordinates"])

    def test_imported_bb_core_is_canonical(self):
        action.check_imported_bb_core()
        self.assertEqual(Path(action.bb_core_module.__file__).resolve(), action.BB_CORE.resolve())


class DurableWalTests(unittest.TestCase):
    def setUp(self):
        self.header = {"schema": action.WAL_SCHEMA, "test": "zero-openmm"}
        self.coordinate = np.arange(action.DIMENSION, dtype=np.float64)
        self.force = np.linspace(-1.0, 1.0, action.DIMENSION, dtype=np.float64)
        self.attempt_id = "ir_internal:00:attempt001"

    def append_json(self, path: Path, record: dict) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def make_closed_wal(self, path: Path) -> None:
        wal = action.DurableWal(path, self.header)
        base = {
            "attempt_id": self.attempt_id,
            "logical_index": 0,
            "original_image_index": 1,
            "coordinate_sha256": action.sha256_array(self.coordinate),
        }
        wal.append({"event": "attempt_started", "utc": "test", **base})
        wal.append(action._success_event(0, self.coordinate, 4.25, self.force, self.attempt_id))
        wal.close()

    def test_full_force_round_trip_and_exact_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "joint.wal.jsonl"
            self.make_closed_wal(path)
            state = action.read_wal(path, self.header)
            self.assertEqual(state["attempt_started"], 1)
            self.assertEqual(state["attempt_succeeded"], 1)
            decoded = action._decode_force(state["successes"][0])
            np.testing.assert_array_equal(decoded, self.force)

    def test_force_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "joint.wal.jsonl"
            self.make_closed_wal(path)
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            raw = bytearray(base64.b64decode(records[-1]["force_le_f8_base64"]))
            raw[0] ^= 1
            records[-1]["force_le_f8_base64"] = base64.b64encode(bytes(raw)).decode("ascii")
            path.write_text("\n".join(json.dumps(item, sort_keys=True) for item in records) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "force SHA-256 mismatch"):
                action.read_wal(path, self.header)

    def test_open_attempt_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "joint.wal.jsonl"
            wal = action.DurableWal(path, self.header)
            wal.append(
                {
                    "event": "attempt_started",
                    "utc": "test",
                    "attempt_id": self.attempt_id,
                    "logical_index": 0,
                    "original_image_index": 1,
                    "coordinate_sha256": action.sha256_array(self.coordinate),
                }
            )
            wal.close()
            with self.assertRaisesRegex(ValueError, "separately versioned"):
                action.read_wal(path, self.header)

    def test_reusable_prefix_is_bound_before_calls(self):
        event = action._success_event(0, self.coordinate, 4.25, self.force, self.attempt_id)
        state = {"successes": {0: event}}
        path = np.zeros((32, action.DIMENSION), dtype=np.float64)
        path[1] = self.coordinate
        action.validate_success_prefix(state, path)
        path[1, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "WAL/source coordinate mismatch"):
            action.validate_success_prefix(state, path)


class WorkflowStaticSafetyTests(unittest.TestCase):
    def test_nonphysical_scripts_do_not_construct_backend(self):
        for path in (FREEZER, AUTHORIZER, PLOTTER):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("OpenMMBackend(", source, str(path))
            self.assertNotIn("evaluate_joint(", source, str(path))

    def test_plotter_has_no_resampling_operator(self):
        source = PLOTTER.read_text(encoding="utf-8")
        for forbidden in ("np.interp(", "CubicSpline(", "interp1d(", "np.linspace("):
            self.assertNotIn(forbidden, source)
        self.assertIn("axis.plot(", source)
        self.assertIn("number_of_original_images_per_path", source)

    def test_physical_call_sites_are_single_and_joint(self):
        source = EVALUATOR.read_text(encoding="utf-8")
        self.assertEqual(source.count("backend.evaluate_joint(coordinate)"), 1)
        self.assertNotIn("evaluate_energy", source)
        self.assertNotIn("evaluate_force", source)

    def test_formula_identifier_and_text_are_stable(self):
        self.assertEqual(action.FORMULA_ID, "original32-segment-trapezoid-int-norm-gradV-dell-v1")
        self.assertIn("0.5*(||F_k||_2+||F_{k+1}||_2)", action.FORMULA_TEXT)
        self.assertEqual(action.N_IMAGES, 32)
        self.assertEqual(action.N_INTERNAL, 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
