from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib

import numpy as np


CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_ROOT))

from src.final_audit_core import (
    _audit_receipt,
    _domain_array_digest,
    _expected_result_identity,
    audit_compact_spring_replay,
    audit_nof,
    count_duplicate_charged_coordinates,
    hj_improved_tangents,
    locf_indices,
    q10_energy_gradient,
    q10_energy_gradient_batch,
    strict_checkpoint_comparison,
    tangent_transcript_sha256,
)
from src.finalize_figure_qa import (
    FigureQAFinalizationError,
    VISUAL_QA_CONFIRMATION,
    finalize_visual_qa,
)
from src.formal_io import canonical_json_bytes, sha256_file
from src.plot_comparison import generate_figure
from src.optimizer_interface import FrozenControls
from src.registered_adapter import TangentEvent, _array_sha256, _tangent_transcript_sha256
from src.result_schema import coordinate_sha256, tangent_transcript_sha256 as schema_transcript
from src.run_frozen_reproduction import _receipt as runner_receipt


def _write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_test_png(path: Path, width: int = 3, height: int = 2) -> None:
    """Write a small, fully CRC-valid RGB PNG without an imaging dependency."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind)
        checksum = zlib.crc32(data, checksum) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    # Filter byte plus one deterministic RGB scanline per row.
    scanlines = b"".join(
        b"\x00" + bytes(((row + column) % 256 for column in range(width * 3)))
        for row in range(height)
    )
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


class IndependentNumericsTests(unittest.TestCase):
    def test_potential_gradient_matches_central_finite_difference(self) -> None:
        q = np.eye(10, dtype=np.float64)
        point = np.asarray([-0.3, 0.8, 0.1, -0.2, 0.05, 0.02, 0.0, 0.0, 0.0, 0.0])
        _, gradient = q10_energy_gradient(point, q)
        step = 1.0e-6
        numerical = np.empty(10)
        for coordinate in range(10):
            plus = point.copy()
            minus = point.copy()
            plus[coordinate] += step
            minus[coordinate] -= step
            numerical[coordinate] = (
                q10_energy_gradient(plus, q)[0] - q10_energy_gradient(minus, q)[0]
            ) / (2.0 * step)
        np.testing.assert_allclose(gradient, numerical, rtol=2.0e-8, atol=2.0e-8)

    def test_locf_and_strict_rules(self) -> None:
        indices = locf_indices(np.asarray([10, 30, 50]), np.asarray([20, 30, 40, 60]))
        np.testing.assert_array_equal(indices, [0, 1, 1, 2])
        comparison = strict_checkpoint_comparison(
            np.asarray([1.0, 2.0]),
            np.asarray([2.0, 2.0]),
            np.asarray([67.4, 67.6]),
            np.asarray([67.2, 67.8]),
        )
        np.testing.assert_array_equal(comparison["i_perp_strict"], [True, False])
        np.testing.assert_array_equal(comparison["joint_strict"], [True, False])


class CompactTangentContractTests(unittest.TestCase):
    @staticmethod
    def _spring_archive(
        *,
        raw_differs_from_source: bool,
        evaluation_nof: tuple[int, int] = (1, 2),
        nof_at_entry: int = 2,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        q = np.eye(10, dtype=np.float64)
        source = np.zeros((4, 10), dtype=np.float64)
        source[:, 0] = np.linspace(-0.4, 0.5, 4)
        raw = source.copy()
        if raw_differs_from_source:
            # This moves the raw proposal enough that its physical energies do
            # not match the frozen source energies, while retaining a one-step
            # (already converged) spring replay.
            raw[1:-1, 2] = 1.0e-3
        energy = q10_energy_gradient_batch(source, q)[0]
        tangent = hj_improved_tangents(raw, energy)
        lengths = np.linalg.norm(np.diff(raw, axis=0), axis=1)
        spring_force = (lengths[1:] - lengths[:-1])[:, None] * tangent
        residual = float(np.max(np.linalg.norm(spring_force, axis=1)))
        if residual > 1.0e-5:
            raise AssertionError("unit spring is not converged in one iteration")
        path_digest = _array_sha256(raw, "tangent-input-path")
        energy_digest = _array_sha256(energy, "tangent-input-energy")
        output_digest = _array_sha256(tangent, "tangent-output")
        event = TangentEvent(1, "spring", 1, path_digest, energy_digest, output_digest)
        transcript = _tangent_transcript_sha256([event])
        archive = {
            "public_nf": np.asarray([2], dtype=np.int64),
            "public_path_history": source[None, :, :],
            "public_energy_history": energy[None, :],
            "recovery_state_nf": np.asarray([], dtype=np.int64),
            "recovery_anchor_path": np.empty((0, 4, 10), dtype=np.float64),
            "recovery_candidate_path": np.empty((0, 4, 10), dtype=np.float64),
            "recovery_public_before_path": np.empty((0, 4, 10), dtype=np.float64),
            "evaluation_nof": np.asarray(evaluation_nof, dtype=np.int64),
            "evaluation_outer": np.asarray([-1, -1], dtype=np.int64),
            "evaluation_phase": np.asarray(["initial_band", "initial_band"], dtype="U"),
            "evaluation_image_index": np.asarray([1, 2], dtype=np.int64),
            "evaluation_coordinate": source[1:-1].copy(),
            "evaluation_energy": energy[1:-1].copy(),
            "reparam_event_id": np.asarray([1], dtype=np.int64),
            "reparam_caller": np.asarray(["unit_spring"], dtype="U"),
            "reparam_outer": np.asarray([1], dtype=np.int64),
            "reparam_nof_at_entry": np.asarray([nof_at_entry], dtype=np.int64),
            "reparam_input_path": raw[None, :, :],
            "reparam_fixed_energy": energy[None, :],
            "reparam_output_path": raw[None, :, :],
            "reparam_iterations": np.asarray([1], dtype=np.int64),
            "reparam_final_residual": np.asarray([residual], dtype=np.float64),
            "reparam_tangent_call_start": np.asarray([0], dtype=np.int64),
            "reparam_tangent_call_count": np.asarray([1], dtype=np.int64),
            "tangent_event_id": np.asarray([1], dtype=np.int64),
            "tangent_parent_reparam_event_id": np.asarray([1], dtype=np.int64),
            "tangent_context_names": np.asarray(["spring"], dtype="U"),
            "tangent_context_code": np.asarray([0], dtype=np.int16),
            "tangent_context_counts": np.asarray([1], dtype=np.int64),
            "tangent_mismatch_count": np.asarray(0, dtype=np.int64),
            "tangent_call_count": np.asarray(1, dtype=np.int64),
            "tangent_reparam_call_count": np.asarray(1, dtype=np.int64),
            "tangent_path_sha256": np.frombuffer(path_digest, dtype=np.uint8)[None, :],
            "tangent_energy_sha256": np.frombuffer(energy_digest, dtype=np.uint8)[None, :],
            "tangent_output_sha256": np.frombuffer(output_digest, dtype=np.uint8)[None, :],
            "tangent_transcript_sha256": np.asarray(transcript, dtype="U"),
        }
        return archive, q, source, raw

    def test_v2_array_and_transcript_digests_match_adapter_and_schema(self) -> None:
        path = np.arange(40, dtype=np.float64).reshape(4, 10) / 17.0
        energy = np.asarray([0.0, 1.0, 3.0, 2.0], dtype=np.float64)
        tangent = hj_improved_tangents(path, energy)
        path_digest = _array_sha256(path, "tangent-input-path")
        energy_digest = _array_sha256(energy, "tangent-input-energy")
        output_digest = _array_sha256(tangent, "tangent-output")
        self.assertEqual(path_digest, _domain_array_digest(path, domain="tangent-input-path"))
        event = TangentEvent(1, "unit", -1, path_digest, energy_digest, output_digest)
        adapter_transcript = _tangent_transcript_sha256([event])
        mapping = {
            "tangent_event_id": np.asarray([1], dtype=np.int64),
            "tangent_parent_reparam_event_id": np.asarray([-1], dtype=np.int64),
            "tangent_context_names": np.asarray(["unit"], dtype="U"),
            "tangent_context_code": np.asarray([0], dtype=np.int16),
            "tangent_context_counts": np.asarray([1], dtype=np.int64),
            "tangent_mismatch_count": np.asarray(0, dtype=np.int64),
            "tangent_path_sha256": np.frombuffer(path_digest, dtype=np.uint8)[None, :],
            "tangent_energy_sha256": np.frombuffer(energy_digest, dtype=np.uint8)[None, :],
            "tangent_output_sha256": np.frombuffer(output_digest, dtype=np.uint8)[None, :],
        }
        self.assertEqual(tangent_transcript_sha256(mapping), adapter_transcript)
        self.assertEqual(schema_transcript(mapping), adapter_transcript)

    def test_one_equal_spacing_spring_event_replays_bitwise(self) -> None:
        archive, q, _, _ = self._spring_archive(raw_differs_from_source=False)
        report, rows = audit_compact_spring_replay(archive, q)
        self.assertEqual(report["spring_tangent_events_independently_replayed"], 1)
        self.assertEqual(report["non_spring_tangent_events_hash_only"], 0)
        self.assertTrue(rows[0]["replay_pass"])

    def test_raw_spring_path_can_differ_from_authenticated_energy_source(self) -> None:
        archive, q, source, raw = self._spring_archive(raw_differs_from_source=True)
        self.assertFalse(np.array_equal(source, raw))
        # This is the v1-auditor failure mode: frozen energies belong to the
        # already evaluated source state, not to this raw normal proposal.
        self.assertGreater(
            float(np.max(np.abs(
                q10_energy_gradient_batch(raw, q)[0]
                - archive["reparam_fixed_energy"][0]
            ))),
            2.0e-11,
        )
        report, rows = audit_compact_spring_replay(archive, q)
        self.assertEqual(report["fixed_energy_sources_authenticated"], 1)
        self.assertEqual(report["events_with_distinct_input_and_energy_source"], 1)
        self.assertEqual(rows[0]["energy_source_kind"], "public_state")
        self.assertEqual(rows[0]["energy_source_index"], 0)
        self.assertEqual(rows[0]["energy_source_nof"], 2)
        self.assertFalse(rows[0]["energy_source_path_equals_reparam_input"])

    def test_source_without_prior_exact_paid_coordinate_is_rejected(self) -> None:
        archive, q, _, _ = self._spring_archive(raw_differs_from_source=True)
        archive["evaluation_coordinate"] = archive["evaluation_coordinate"].copy()
        archive["evaluation_coordinate"][1, 4] = 1.0
        with self.assertRaisesRegex(Exception, "no causal, fully paid source state"):
            audit_compact_spring_replay(archive, q)

    def test_future_paid_evaluation_cannot_authenticate_earlier_spring(self) -> None:
        archive, q, _, _ = self._spring_archive(
            raw_differs_from_source=True,
            evaluation_nof=(3, 4),
            nof_at_entry=2,
        )
        with self.assertRaisesRegex(Exception, "at or before NoF 2"):
            audit_compact_spring_replay(archive, q)

    def test_complete_paid_band_reports_phase_and_is_not_an_arbitrary_window(self) -> None:
        archive, q, _, _ = self._spring_archive(raw_differs_from_source=True)
        archive["public_nf"] = np.asarray([3], dtype=np.int64)
        archive["evaluation_phase"] = np.asarray(
            ["direct_spring_check", "direct_spring_check"], dtype="U"
        )
        report, rows = audit_compact_spring_replay(archive, q)
        self.assertEqual(rows[0]["energy_source_kind"], "paid_complete_band")
        self.assertEqual(rows[0]["energy_source_phase"], "direct_spring_check")
        self.assertEqual(
            report["fixed_energy_source_phase_counts"], {"direct_spring_check": 1}
        )

        split_phase = dict(archive)
        split_phase["evaluation_phase"] = np.asarray(
            ["direct_spring_check", "recovery_band_trial"], dtype="U"
        )
        with self.assertRaisesRegex(Exception, "no causal, fully paid source state"):
            audit_compact_spring_replay(split_phase, q)


class IdentityReceiptContractTests(unittest.TestCase):
    def test_instrumentation_identity_paths_and_runner_v2_receipt(self) -> None:
        digests = {name: hashlib.sha256(name.encode("ascii")).hexdigest() for name in (
            "algorithm", "adapter", "runtime", "backend", "runner", "exploratory",
            "preformal", "instrumentation",
        )}
        source = {
            "src/q10_unified_cycle_optimizer.py": digests["algorithm"],
            "src/registered_adapter.py": digests["adapter"],
            "src/run_frozen_reproduction.py": digests["runner"],
            "evidence/exploratory_candidate.py": digests["exploratory"],
            "evidence/preformal_equivalence.json": digests["preformal"],
            "evidence/preformal_instrumentation_equivalence.json": digests["instrumentation"],
        }
        dependency = {
            "repo:core.py": digests["runtime"],
            "repo:backend.py": digests["backend"],
        }
        paths = {
            "algorithm_source_sha256": "campaign:src/q10_unified_cycle_optimizer.py",
            "optimizer_sha256": "campaign:src/q10_unified_cycle_optimizer.py",
            "adapter_sha256": "campaign:src/registered_adapter.py",
            "runtime_sha256": "repo:core.py",
            "backend_sha256": "repo:backend.py",
            "runner_sha256": "campaign:src/run_frozen_reproduction.py",
            "exploratory_source_sha256": "campaign:evidence/exploratory_candidate.py",
            "preformal_equivalence_sha256": "campaign:evidence/preformal_equivalence.json",
            "instrumentation_equivalence_sha256": (
                "campaign:evidence/preformal_instrumentation_equivalence.json"
            ),
        }
        identity = {
            "algorithm_source_sha256": digests["algorithm"],
            "optimizer_sha256": digests["algorithm"],
            "adapter_sha256": digests["adapter"],
            "runtime_sha256": digests["runtime"],
            "backend_sha256": digests["backend"],
            "runner_sha256": digests["runner"],
            "exploratory_source_sha256": digests["exploratory"],
            "preformal_equivalence_sha256": digests["preformal"],
            "instrumentation_equivalence_sha256": digests["instrumentation"],
            "q_sha256": "372962a47504e6df28a20d8b022482a6835680765434b4385a6dba0e826974bc",
        }
        registration = {
            "campaign_id": "test",
            "source_sha256": source,
            "dependency_sha256": dependency,
            "result_identity_paths": paths,
            "result_identity": identity,
        }
        resolved = _expected_result_identity(registration)
        self.assertEqual(
            resolved["instrumentation_equivalence_sha256"], digests["instrumentation"]
        )
        controls = FrozenControls(10, 0.003, "3e-3", 0.1, 7, 3000)
        registration_hash = "a" * 64
        seal_hash = "b" * 64
        runtime_identity = {"python_version": "3.8.20"}
        bundle = {
            "registration": registration,
            "registration_sha256": registration_hash,
            "registration_seal_sha256": seal_hash,
            "result_identity": identity,
            "runtime_identity": runtime_identity,
        }
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            result = campaign / "runs/n10/result.npz"
            result.parent.mkdir(parents=True)
            result.write_bytes(b"synthetic-result")
            receipt_payload = runner_receipt(
                bundle, controls, "runs/n10/result.npz", sha256_file(result),
                {"terminal_nf": 782, "stop_reason": "stationary"},
                runtime_identity,
            )
            receipt_path = campaign / "reports/receipts/n10/result.receipt.json"
            receipt_path.parent.mkdir(parents=True)
            receipt_path.write_bytes(canonical_json_bytes(receipt_payload))
            scalar = lambda value: np.asarray(value, dtype="U")
            archive = {
                "campaign_id": scalar("test"), "run_id": scalar(controls.run_id),
                "algorithm_source_sha256": scalar(identity["algorithm_source_sha256"]),
                "optimizer_sha256": scalar(identity["optimizer_sha256"]),
                "adapter_sha256": scalar(identity["adapter_sha256"]),
                "runtime_sha256": scalar(identity["runtime_sha256"]),
                "backend_sha256": scalar(identity["backend_sha256"]),
                "runner_sha256": scalar(identity["runner_sha256"]),
                "exploratory_source_sha256": scalar(identity["exploratory_source_sha256"]),
                "preformal_equivalence_sha256": scalar(identity["preformal_equivalence_sha256"]),
                "instrumentation_equivalence_sha256": scalar(
                    identity["instrumentation_equivalence_sha256"]
                ),
                "q_sha256": scalar(identity["q_sha256"]),
                "runtime_identity_json": scalar(json.dumps(runtime_identity, sort_keys=True, separators=(",", ":"))),
                "terminal_nof": np.asarray(782, dtype=np.int64),
                "stop_reason": scalar("stationary"),
            }
            checked = _audit_receipt(
                receipt_path, result, archive, 10, campaign, registration_hash,
                seal_hash, registration,
            )
            self.assertEqual(checked["schema_version"], 2)


class NoFTests(unittest.TestCase):
    def test_exact_duplicate_is_per_image_and_full_ledger_closes(self) -> None:
        coordinate = np.arange(10, dtype=np.float64)
        archive = {
            "terminal_nof": np.asarray(2, dtype=np.int64),
            "evaluation_sequence": np.asarray([1, 2], dtype=np.int64),
            "evaluation_nof": np.asarray([1, 2], dtype=np.int64),
            "evaluation_image_index": np.asarray([1, 2], dtype=np.int64),
            "evaluation_phase": np.asarray(["initial_band", "initial_band"], dtype="U"),
            "evaluation_coordinate": np.asarray([coordinate, coordinate]),
            "evaluation_coordinate_sha256": np.asarray(
                [coordinate_sha256(coordinate), coordinate_sha256(coordinate)], dtype="U"
            ),
            "optimizer_evaluation_trace_json": np.asarray([
                json.dumps({
                    "sequence": 1, "article_nf": 1, "image_index": 1,
                    "coordinate_sha256": coordinate_sha256(coordinate),
                }, sort_keys=True, separators=(",", ":")),
                json.dumps({
                    "sequence": 2, "article_nf": 2, "image_index": 2,
                    "coordinate_sha256": coordinate_sha256(coordinate),
                }, sort_keys=True, separators=(",", ":")),
            ], dtype="U"),
            "per_image_joint_evaluations": np.asarray([0, 1, 1, 0], dtype=np.int64),
            "nof_category_names": np.asarray(["initial_band"], dtype="U"),
            "nof_category_counts": np.asarray([2], dtype=np.int64),
            "blocked_uncached_misses": np.asarray(0, dtype=np.int64),
        }
        self.assertEqual(count_duplicate_charged_coordinates(archive), 0)
        self.assertEqual(audit_nof(archive)["trace_length"], 2)
        archive["evaluation_image_index"] = np.asarray([1, 1], dtype=np.int64)
        self.assertEqual(count_duplicate_charged_coordinates(archive), 1)


class PlotTests(unittest.TestCase):
    def test_authenticated_2x2_and_provenance(self) -> None:
        # Keep this scratch tree under the real campaign so the finalizer can
        # authenticate its own and the plotter's source paths without a test-
        # only bypass. TemporaryDirectory removes every artifact afterwards.
        with tempfile.TemporaryDirectory(dir=CAMPAIGN_ROOT) as directory:
            root = Path(directory)
            audit_dir = root / "audit"
            figures = root / "figures"
            audit_dir.mkdir()
            figures.mkdir()
            checkpoint_rows = []
            trajectory_rows = []
            terminals = {10: 951, 20: 2377}
            counts = {10: 79, 20: 98}
            for n_internal in (10, 20):
                checkpoints = np.linspace(n_internal + 1, terminals[n_internal], counts[n_internal], dtype=int)
                checkpoints = np.maximum.accumulate(checkpoints)
                # The ranges are long enough that the integer grid is strict.
                for index, checkpoint in enumerate(checkpoints, 1):
                    checkpoint_rows.append({
                        "n_internal": n_internal,
                        "checkpoint_index": index,
                        "checkpoint_nof": int(checkpoint),
                        "i_perp_strict": "true",
                        "barrier_strict": "true",
                        "joint_strict": "true",
                    })
                trajectory_rows.extend([
                    {
                        "n_internal": n_internal, "method": "Article IVM",
                        "nof": n_internal, "i_perp": 10.0, "delta_e": 68.0,
                        "barrier_error": 0.5, "is_locf_extension": "false",
                        "source_state_index": 0,
                    },
                    {
                        "n_internal": n_internal, "method": "Article IVM",
                        "nof": terminals[n_internal], "i_perp": 1.0,
                        "delta_e": 67.7, "barrier_error": 0.2,
                        "is_locf_extension": "false", "source_state_index": 1,
                    },
                    {
                        "n_internal": n_internal, "method": "IR-IVM",
                        "nof": n_internal, "i_perp": 9.0, "delta_e": 67.9,
                        "barrier_error": 0.4, "is_locf_extension": "false",
                        "source_state_index": 0,
                    },
                    {
                        "n_internal": n_internal, "method": "IR-IVM",
                        "nof": (782 if n_internal == 10 else terminals[n_internal]),
                        "i_perp": 0.5, "delta_e": 67.6, "barrier_error": 0.1,
                        "is_locf_extension": "false", "source_state_index": 1,
                    },
                ])
                if n_internal == 10:
                    trajectory_rows.append({
                        "n_internal": 10, "method": "IR-IVM", "nof": 951,
                        "i_perp": 0.5, "delta_e": 67.6, "barrier_error": 0.1,
                        "is_locf_extension": "true", "source_state_index": 1,
                    })
            checkpoint_path = audit_dir / "checkpoint_comparison.csv"
            trajectory_path = audit_dir / "public_trajectory_metrics.csv"
            _write_csv(checkpoint_path, list(checkpoint_rows[0]), checkpoint_rows)
            _write_csv(trajectory_path, list(trajectory_rows[0]), trajectory_rows)
            zero = "0" * 64

            registration_path = root / "registrations/CAMPAIGN_PREREGISTRATION.json"
            seal_path = root / "registrations/CAMPAIGN_PREREGISTRATION.sha256"
            registration_path.parent.mkdir()
            registration_path.write_bytes(canonical_json_bytes({"campaign": "unit-test"}))
            seal_path.write_text(sha256_file(registration_path) + "\n", encoding="ascii")
            run_specs = []
            for n_internal in (10, 20):
                result_path = root / f"runs/n{n_internal}/result.npz"
                receipt_path = root / f"reports/receipts/n{n_internal}/result.receipt.json"
                result_path.parent.mkdir(parents=True)
                receipt_path.parent.mkdir(parents=True)
                result_path.write_bytes(f"synthetic-result-{n_internal}".encode("ascii"))
                receipt_path.write_bytes(
                    canonical_json_bytes({"n_internal": n_internal, "synthetic": True})
                )
                run_specs.append({
                    "n_internal": n_internal,
                    "result_path": result_path.relative_to(CAMPAIGN_ROOT).as_posix(),
                    "result_sha256": sha256_file(result_path),
                    "receipt_path": receipt_path.relative_to(CAMPAIGN_ROOT).as_posix(),
                    "receipt_sha256": sha256_file(receipt_path),
                })
            manifest = {
                "outcome": "complete_scientific_pass",
                "integrity_complete": True,
                "audit_source_sha256": zero,
                "source_sha256": {
                    "src/plot_comparison.py": sha256_file(
                        CAMPAIGN_ROOT / "src/plot_comparison.py"
                    ),
                    "src/finalize_figure_qa.py": sha256_file(
                        CAMPAIGN_ROOT / "src/finalize_figure_qa.py"
                    ),
                },
                "registration": {
                    "relative_path": registration_path.relative_to(CAMPAIGN_ROOT).as_posix(),
                    "sha256": sha256_file(registration_path),
                    "seal_relative_path": seal_path.relative_to(CAMPAIGN_ROOT).as_posix(),
                    "seal_sha256": sha256_file(seal_path),
                },
                "registered_controls": {
                    "10": {"alpha_label": "3e-3", "r": 0.1, "m_max": 7},
                    "20": {"alpha_label": "1e-3", "r": 0.3, "m_max": 8},
                },
                "runs": run_specs,
                "artifacts": {
                    checkpoint_path.name: {
                        "relative_path": checkpoint_path.name,
                        "sha256": sha256_file(checkpoint_path),
                    },
                    trajectory_path.name: {
                        "relative_path": trajectory_path.name,
                        "sha256": sha256_file(trajectory_path),
                    },
                },
            }
            audit_manifest = audit_dir / "audit_manifest.json"
            audit_manifest.write_bytes(canonical_json_bytes(manifest))
            pdf = figures / "comparison.pdf"
            png = figures / "comparison.png"
            figure_manifest = figures / "figure_manifest.json"
            provenance = figures / "FIGURE_PROVENANCE.md"
            generate_figure(audit_manifest, pdf, png, figure_manifest, provenance)
            self.assertGreater(pdf.stat().st_size, 1000)
            self.assertGreater(png.stat().st_size, 1000)
            self.assertTrue(provenance.is_file())
            rendered = json.loads(figure_manifest.read_text(encoding="utf-8"))
            self.assertTrue(rendered["cryptographic_provenance_complete"])
            self.assertEqual(rendered["checkpoint_counts"], {"10": 79, "20": 98})

            qa_png = figures / "poppler-comparison-1.png"
            _write_test_png(qa_png, width=640, height=480)
            final_manifest = figures / "final_figure_manifest.json"

            with self.assertRaises(FigureQAFinalizationError):
                finalize_visual_qa(
                    figure_manifest,
                    audit_manifest,
                    qa_png,
                    final_manifest,
                    campaign_root=CAMPAIGN_ROOT,
                    confirmation="yes",
                    reviewer="unit test reviewer",
                )
            self.assertFalse(final_manifest.exists())

            # A copied/direct plotting PNG is not acceptable as independent
            # evidence of the rendered PDF.
            with self.assertRaises(FigureQAFinalizationError):
                finalize_visual_qa(
                    figure_manifest,
                    audit_manifest,
                    png,
                    final_manifest,
                    campaign_root=CAMPAIGN_ROOT,
                    confirmation=VISUAL_QA_CONFIRMATION,
                    reviewer="unit test reviewer",
                )
            self.assertFalse(final_manifest.exists())

            original_checkpoint = checkpoint_path.read_bytes()
            checkpoint_path.write_bytes(original_checkpoint + b"drift")
            with self.assertRaises(FigureQAFinalizationError):
                finalize_visual_qa(
                    figure_manifest,
                    audit_manifest,
                    qa_png,
                    final_manifest,
                    campaign_root=CAMPAIGN_ROOT,
                    confirmation=VISUAL_QA_CONFIRMATION,
                    reviewer="unit test reviewer",
                )
            self.assertFalse(final_manifest.exists())
            checkpoint_path.write_bytes(original_checkpoint)

            original_pdf = pdf.read_bytes()
            pdf.write_bytes(original_pdf + b"drift")
            with self.assertRaises(FigureQAFinalizationError):
                finalize_visual_qa(
                    figure_manifest,
                    audit_manifest,
                    qa_png,
                    final_manifest,
                    campaign_root=CAMPAIGN_ROOT,
                    confirmation=VISUAL_QA_CONFIRMATION,
                    reviewer="unit test reviewer",
                )
            self.assertFalse(final_manifest.exists())
            pdf.write_bytes(original_pdf)

            finalized_path = finalize_visual_qa(
                figure_manifest,
                audit_manifest,
                qa_png,
                final_manifest,
                campaign_root=CAMPAIGN_ROOT,
                confirmation=VISUAL_QA_CONFIRMATION,
                reviewer="unit test reviewer",
            )
            self.assertEqual(finalized_path, final_manifest.resolve())
            final = json.loads(final_manifest.read_text(encoding="utf-8"))
            self.assertEqual(final["status"], "complete")
            self.assertEqual(final["outcome"], "complete_visual_qa_pass")
            self.assertTrue(final["formal_provenance_complete"])
            self.assertFalse(final["scientific_results_modified"])
            self.assertEqual(
                final["artifacts"]["poppler_rendered_qa_png"]["sha256"],
                sha256_file(qa_png),
            )
            self.assertEqual(
                final["artifacts"]["pdf"]["sha256"], sha256_file(pdf)
            )
            self.assertEqual(
                final["artifacts"]["png"]["sha256"], sha256_file(png)
            )
            self.assertEqual(
                final["artifacts"]["provenance"]["sha256"],
                sha256_file(provenance),
            )
            self.assertEqual(
                final["audit_manifest"]["sha256"], sha256_file(audit_manifest)
            )

            # Completion is one-shot: a second invocation cannot overwrite
            # the canonical final manifest.
            with self.assertRaises(FigureQAFinalizationError):
                finalize_visual_qa(
                    figure_manifest,
                    audit_manifest,
                    qa_png,
                    final_manifest,
                    campaign_root=CAMPAIGN_ROOT,
                    confirmation=VISUAL_QA_CONFIRMATION,
                    reviewer="unit test reviewer",
                )


if __name__ == "__main__":
    unittest.main()
