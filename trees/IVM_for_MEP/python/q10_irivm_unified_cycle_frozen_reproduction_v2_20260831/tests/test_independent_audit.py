from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_ROOT))

from src.audit_results import (
    IndependentAuditError,
    audit_spring_and_tangent_events,
    count_duplicate_charged_coordinates,
    hj_improved_tangents,
    locf_indices,
    replay_spring_event,
    strict_checkpoint_comparison,
)
from src.formal_io import (
    FormalIOError,
    HostLocalFcntlLock,
    atomic_write_bytes,
    canonical_json_bytes,
    load_canonical_json,
    resolve_registered_path,
)
from src.result_schema import (
    EXPECTED_NOF_CATEGORIES,
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    ResultSchemaError,
    coordinate_sha256,
    q_matrix_sha256,
    validate_result_mapping,
)


def unicode_scalar(value: str) -> np.ndarray:
    return np.asarray(value, dtype="U")


def valid_minimal_result() -> dict[str, np.ndarray]:
    n = 10
    total = n + 2
    terminal = n
    path = np.zeros((total, 10), dtype=np.float64)
    path[:, 0] = np.linspace(0.0, 1.0, total)
    coordinates = path[1:-1].copy()
    sha = "0" * 64
    q_matrix = np.eye(10, dtype=np.float64)
    q_sha = q_matrix_sha256(q_matrix)
    transcript_sha = hashlib.sha256(
        b"q10-tangent-transcript-v2\x00"
    ).hexdigest()
    coordinate_hashes = [coordinate_sha256(row) for row in coordinates]
    optimizer_trace = np.asarray(
        [
            json.dumps(
                {
                    "sequence": index + 1,
                    "article_nf": index + 1,
                    "outer": -1,
                    "phase": "unassigned",
                    "image_index": index + 1,
                    "coordinate_sha256": coordinate_hashes[index],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for index in range(terminal)
        ],
        dtype="U",
    )
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "campaign_id": unicode_scalar("test-campaign"),
        "run_id": unicode_scalar("n10"),
        "config_json": unicode_scalar(json.dumps({"n_internal": n}, sort_keys=True)),
        "registration_sha256": unicode_scalar(sha),
        "registration_seal_sha256": unicode_scalar(sha),
        "algorithm_source_sha256": unicode_scalar(sha),
        "optimizer_sha256": unicode_scalar(sha),
        "adapter_sha256": unicode_scalar(sha),
        "runtime_sha256": unicode_scalar(sha),
        "backend_sha256": unicode_scalar(sha),
        "runner_sha256": unicode_scalar(sha),
        "dependency_manifest_json": unicode_scalar("{}"),
        "exploratory_source_sha256": unicode_scalar(sha),
        "preformal_equivalence_sha256": unicode_scalar(sha),
        "instrumentation_equivalence_sha256": unicode_scalar(sha),
        "runtime_identity_json": unicode_scalar("{}"),
        "optimizer_summary_json": unicode_scalar("{}"),
        "q_matrix": q_matrix,
        "q_sha256": unicode_scalar(q_sha),
        "tangent_definition": unicode_scalar("HJ improved tangent"),
        "reparameterization_definition": unicode_scalar("nearest-neighbour HJ spring flow"),
        "tangent_transcript_sha256": unicode_scalar(transcript_sha),
        "spring_dtau": np.asarray(0.05, dtype=np.float64),
        "spring_max_iter": np.asarray(500, dtype=np.int64),
        "spring_tolerance": np.asarray(1.0e-5, dtype=np.float64),
        "stop_reason": unicode_scalar("stationary"),
        "failure_message": unicode_scalar(""),
        "budget": np.asarray(3000, dtype=np.int64),
        "terminal_nof": np.asarray(terminal, dtype=np.int64),
        "public_nf": np.asarray([terminal], dtype=np.int64),
        "public_outer": np.asarray([-1], dtype=np.int64),
        "public_phase": np.asarray(["initial"], dtype="U"),
        "public_path_history": path[None, :, :],
        "public_energy_history": np.zeros((1, total), dtype=np.float64),
        "public_gradient_history": np.zeros((1, total, 10), dtype=np.float64),
        "public_state_changed": np.asarray([True], dtype=np.bool_),
        "working_equals_public_at_exit": np.asarray(True, dtype=np.bool_),
        "evaluation_sequence": np.arange(1, terminal + 1, dtype=np.int64),
        "evaluation_nof": np.arange(1, terminal + 1, dtype=np.int64),
        "evaluation_outer": np.full(terminal, -1, dtype=np.int64),
        "evaluation_phase": np.full(terminal, "initial_band", dtype="<U12"),
        "evaluation_image_index": np.arange(1, n + 1, dtype=np.int64),
        "evaluation_coordinate": coordinates,
        "evaluation_coordinate_sha256": np.asarray(coordinate_hashes, dtype="U"),
        "evaluation_energy": np.zeros(terminal, dtype=np.float64),
        "evaluation_gradient": np.zeros((terminal, 10), dtype=np.float64),
        "optimizer_evaluation_trace_json": optimizer_trace,
        "per_image_joint_evaluations": np.asarray([0] + [1] * n + [0], dtype=np.int64),
        "nof_category_names": np.asarray(EXPECTED_NOF_CATEGORIES, dtype="U"),
        "nof_category_counts": np.asarray(
            [terminal, 0, 0, 0, 0], dtype=np.int64
        ),
        "blocked_uncached_misses": np.asarray(0, dtype=np.int64),
        "main_event_json": np.asarray([], dtype="U"),
        "recovery_event_json": np.asarray([], dtype="U"),
        "recovery_state_nf": np.asarray([], dtype=np.int64),
        "recovery_anchor_path": np.empty((0, total, 10), dtype=np.float64),
        "recovery_candidate_path": np.empty((0, total, 10), dtype=np.float64),
        "recovery_public_before_path": np.empty((0, total, 10), dtype=np.float64),
        "reparam_event_id": np.asarray([], dtype=np.int64),
        "reparam_caller": np.asarray([], dtype="U"),
        "reparam_outer": np.asarray([], dtype=np.int64),
        "reparam_nof_at_entry": np.asarray([], dtype=np.int64),
        "reparam_input_path": np.empty((0, total, 10), dtype=np.float64),
        "reparam_fixed_energy": np.empty((0, total), dtype=np.float64),
        "reparam_output_path": np.empty((0, total, 10), dtype=np.float64),
        "reparam_iterations": np.asarray([], dtype=np.int64),
        "reparam_final_residual": np.asarray([], dtype=np.float64),
        "reparam_tangent_call_start": np.asarray([], dtype=np.int64),
        "reparam_tangent_call_count": np.asarray([], dtype=np.int64),
        "tangent_call_count": np.asarray(0, dtype=np.int64),
        "tangent_mismatch_count": np.asarray(0, dtype=np.int64),
        "tangent_reparam_call_count": np.asarray(0, dtype=np.int64),
        "tangent_event_id": np.asarray([], dtype=np.int64),
        "tangent_parent_reparam_event_id": np.asarray([], dtype=np.int64),
        "tangent_context_names": np.asarray([], dtype="U"),
        "tangent_context_code": np.asarray([], dtype=np.int16),
        "tangent_context_counts": np.asarray([], dtype=np.int64),
        "tangent_path_sha256": np.empty((0, 32), dtype=np.uint8),
        "tangent_energy_sha256": np.empty((0, 32), dtype=np.uint8),
        "tangent_output_sha256": np.empty((0, 32), dtype=np.uint8),
        "memory_image_index": np.asarray([], dtype=np.int64),
        "memory_pair_order": np.asarray([], dtype=np.int64),
        "memory_s": np.empty((0, 10), dtype=np.float64),
        "memory_y": np.empty((0, 10), dtype=np.float64),
    }
    assert set(arrays) == REQUIRED_FIELDS
    return arrays


class ResultSchemaTests(unittest.TestCase):
    def test_minimal_pickle_free_result_validates(self) -> None:
        report = validate_result_mapping(valid_minimal_result())
        self.assertEqual(report.terminal_nof, 10)
        self.assertEqual(report.dimensions.n_internal, 10)

    def test_missing_or_extra_field_fails(self) -> None:
        arrays = valid_minimal_result()
        del arrays["q_sha256"]
        with self.assertRaises(ResultSchemaError):
            validate_result_mapping(arrays)
        arrays = valid_minimal_result()
        arrays["unregistered"] = np.asarray(1, dtype=np.int64)
        with self.assertRaises(ResultSchemaError):
            validate_result_mapping(arrays)

    def test_coordinate_hash_and_nof_ledger_fail_closed(self) -> None:
        arrays = valid_minimal_result()
        arrays["evaluation_coordinate_sha256"] = arrays[
            "evaluation_coordinate_sha256"
        ].copy()
        arrays["evaluation_coordinate_sha256"][3] = "f" * 64
        with self.assertRaises(ResultSchemaError):
            validate_result_mapping(arrays)
        arrays = valid_minimal_result()
        arrays["nof_category_counts"] = np.asarray([9], dtype=np.int64)
        with self.assertRaises(ResultSchemaError):
            validate_result_mapping(arrays)


class IndependentGeometryAuditTests(unittest.TestCase):
    def test_hj_monotone_tangent(self) -> None:
        path = np.zeros((4, 10), dtype=np.float64)
        path[:, 0] = [0.0, 0.2, 0.7, 1.0]
        tangents = hj_improved_tangents(path, np.asarray([0.0, 1.0, 2.0, 3.0]))
        expected = np.zeros((2, 10), dtype=np.float64)
        expected[:, 0] = 1.0
        np.testing.assert_array_equal(tangents, expected)

    def test_equal_spacing_spring_event_and_log_replay(self) -> None:
        path = np.zeros((4, 10), dtype=np.float64)
        path[:, 0] = [0.0, 1.0, 2.0, 3.0]
        energies = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        output, residual, tangent_log = replay_spring_event(path, energies, 1)
        self.assertEqual(residual, 0.0)
        np.testing.assert_array_equal(output, path)
        archive = {
            "reparam_event_id": np.asarray([1], dtype=np.int64),
            "reparam_tangent_call_start": np.asarray([0], dtype=np.int64),
            "reparam_tangent_call_count": np.asarray([1], dtype=np.int64),
            "reparam_input_path": path[None, :, :],
            "reparam_fixed_energy": energies[None, :],
            "reparam_output_path": output[None, :, :],
            "reparam_iterations": np.asarray([1], dtype=np.int64),
            "reparam_final_residual": np.asarray([residual], dtype=np.float64),
            "spring_dtau": np.asarray(0.05, dtype=np.float64),
            "spring_tolerance": np.asarray(1.0e-5, dtype=np.float64),
            "tangent_path": np.asarray([tangent_log[0][0]], dtype=np.float64),
            "tangent_energy": energies[None, :],
            "tangent_output": np.asarray([tangent_log[0][1]], dtype=np.float64),
            "tangent_parent_reparam_event_id": np.asarray([1], dtype=np.int64),
        }
        report = audit_spring_and_tangent_events(archive)
        self.assertEqual(report.event_count, 1)
        self.assertEqual(report.tangent_call_count, 1)
        corrupt = dict(archive)
        corrupt["tangent_output"] = archive["tangent_output"].copy()
        corrupt["tangent_output"][0, 0, 0] = -1.0
        with self.assertRaises(IndependentAuditError):
            audit_spring_and_tangent_events(corrupt)

    def test_locf_and_strict_comparison(self) -> None:
        indices = locf_indices(
            np.asarray([10, 30, 50]), np.asarray([20, 30, 40, 60])
        )
        np.testing.assert_array_equal(indices, [0, 1, 1, 2])
        comparison = strict_checkpoint_comparison(
            np.asarray([1.0, 2.0]),
            np.asarray([2.0, 2.0]),
            np.asarray([67.4, 67.6]),
            np.asarray([67.2, 67.8]),
        )
        np.testing.assert_array_equal(comparison["i_perp_strict"], [True, False])
        np.testing.assert_array_equal(comparison["joint_strict"], [True, False])

    def test_exact_coordinate_duplicate_detection_is_per_image(self) -> None:
        coordinate = np.arange(10, dtype=np.float64)
        archive = {
            "evaluation_image_index": np.asarray([1, 2, 1], dtype=np.int64),
            "evaluation_coordinate": np.asarray([coordinate, coordinate, coordinate]),
        }
        self.assertEqual(count_duplicate_charged_coordinates(archive), 1)


class FormalIOTests(unittest.TestCase):
    def test_relative_resolver_rejects_escape_and_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ok.txt").write_text("ok", encoding="utf-8")
            self.assertEqual(resolve_registered_path(root, "ok.txt"), (root / "ok.txt").resolve())
            with self.assertRaises(FormalIOError):
                resolve_registered_path(root, "../escape.txt", must_exist=False)
            with self.assertRaises(FormalIOError):
                resolve_registered_path(root, "/tmp/absolute.txt", must_exist=False)

    def test_atomic_write_refuses_overwrite_and_json_is_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "payload.json"
            atomic_write_bytes(target, canonical_json_bytes({"z": 1, "a": 2}))
            self.assertEqual(load_canonical_json(target), {"a": 2, "z": 1})
            with self.assertRaises(FormalIOError):
                atomic_write_bytes(target, b"replacement")

    def test_host_local_lock_is_nonblocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "campaign.lock"
            with HostLocalFcntlLock(lock_path):
                with self.assertRaises(FormalIOError):
                    with HostLocalFcntlLock(lock_path):
                        self.fail("second lock unexpectedly acquired")


if __name__ == "__main__":
    unittest.main()
