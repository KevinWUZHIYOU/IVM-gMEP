"""Zero-OpenMM contract/preflight tests for IR-IVM joint-E/F v2."""

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


class IvmJointV2PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freezer = load_module("freeze_ivm_joint_v2_test", ROOT / "scripts" / "freeze_bb_dense_ivm_four_joint_v2_contract.py")
        cls.evaluator = load_module("evaluate_ivm_joint_v2_test", ROOT / "scripts" / "evaluate_bb_dense_four_ivm_joint_v2.py")

    def make_verification(self, directory: Path):
        inventory = []
        for history_index, (call, nf) in enumerate(zip(self.evaluator.CALLS, self.evaluator.EXPECTED_NF)):
            base = np.arange(32, dtype=np.float64)[:, None]
            direction = np.linspace(1.0, 2.0, 9396, dtype=np.float64)[None, :]
            path = np.ascontiguousarray(0.001 * base * direction + history_index)
            energy = np.ascontiguousarray(np.sum(path * path, axis=1))
            force = np.ascontiguousarray(-2.0 * path)
            i_perp = float(self.evaluator.ENGINE.path_diagnostics(path, energy, force[1:-1])["i_perp"])
            path_sha = self.evaluator.ENGINE.sha256_array(path)
            energy_sha = self.evaluator.ENGINE.sha256_array(energy)
            force_sha = self.evaluator.ENGINE.sha256_array(force)
            shard = directory / f"online_call_{call}.npz"
            joint = np.zeros(32, dtype=np.int64)
            joint[1] = nf
            inherited_present = history_index % 2 == 1
            source_kind = "live_counter_arrays" if inherited_present else "historical_counter_snapshot"
            schema_keys = [*self.evaluator.COMMON_COUNTER_KEYS]
            unavailable = [self.evaluator.INHERITED_COUNTER_KEY]
            shard_arrays = dict(
                schema=np.asarray("bb-irivm-full-state-history-recorder-v2"), label=np.asarray(self.evaluator.R4_LABEL),
                history_index=np.asarray(history_index), call_per_image=np.asarray(call), optimization_N_F=np.asarray(nf),
                path32_A=path, energy_eV=energy, force_eV_per_A=force,
                path_sha256=np.asarray(path_sha), energy_sha256=np.asarray(energy_sha), force_sha256=np.asarray(force_sha),
                counter_source_kind=np.asarray(source_kind),
                counter_schema_keys=np.asarray(schema_keys, dtype="U64"),
                unavailable_counter_fields=np.asarray(unavailable, dtype="U64"),
                inherited_cache_entries_present=np.asarray(inherited_present),
                joint_evaluation_vector=joint, energy_request_vector=joint, force_request_vector=joint,
                cache_hit_vector=np.zeros(32, dtype=np.int64), total_query_event_vector=joint,
                both_request_vector=joint, budget_blocked_miss_vector=np.zeros(32, dtype=np.int64),
                discrete_I_perp=np.asarray(i_perp), fully_evaluated_current_path=np.asarray(True), provisional_reportable=np.asarray(True),
                terminal_uncommitted_tail=np.asarray(False), additional_physical_calls=np.asarray(0),
            )
            if inherited_present:
                schema_keys.append(self.evaluator.INHERITED_COUNTER_KEY)
                shard_arrays["counter_schema_keys"] = np.asarray(schema_keys, dtype="U64")
                shard_arrays["unavailable_counter_fields"] = np.asarray([], dtype="U64")
                inherited = np.zeros(32, dtype=np.int64)
                inherited[[0, -1]] = 1
                shard_arrays["inherited_cache_entry_vector"] = inherited
            np.savez(shard, **shard_arrays)
            inventory.append({
                "history_index": history_index, "call_per_image": call, "optimization_N_F": nf,
                "discrete_I_perp_eV2_per_A": i_perp, "path_sha256": path_sha, "energy_sha256": energy_sha, "force_sha256": force_sha,
                "shard": str(shard), "shard_sha256": self.evaluator.ENGINE.sha256_file(shard),
                "counter_source_kind": source_kind, "counter_schema_keys": schema_keys,
                "unavailable_counter_fields": [] if inherited_present else unavailable,
                "formally_reportable": True, "terminal_uncommitted_charged_tail": False,
            })
        registration = directory / "canonical_r4_registration.json"
        registration.write_text(json.dumps({"schema": "bb-irivm-selected-full-state-history-output-only-replay-v4", "status": "FROZEN_BEFORE_FIRST_R4_FULL_STATE_HISTORY_REPLAY_PHYSICAL_CALL", "label": self.evaluator.R4_LABEL}), encoding="utf-8")
        authorization = directory / "canonical_r4_authorization.json"
        authorization.write_text(json.dumps({"schema": "bb-irivm-selected-full-state-history-output-only-replay-authorization-v4", "enabled": True, "label": self.evaluator.R4_LABEL}), encoding="utf-8")
        reference = directory / "reference.npz"
        replay = directory / "replay.npz"
        np.savez(reference, marker=np.asarray(1))
        np.savez(replay, marker=np.asarray(1))
        replay_audit = directory / "replay.audit.json"
        replay_audit.write_text(json.dumps({"consistent": True}), encoding="utf-8")
        capture_manifest = directory / "capture.manifest.json"
        capture_manifest.write_text(json.dumps({"schema": "bb-irivm-full-state-history-manifest-v2", "status": "COMPLETE_PENDING_EXACT_PREFIX_VERIFICATION", "label": self.evaluator.R4_LABEL}), encoding="utf-8")
        capture_audit = directory / "capture.audit.json"
        capture_audit.write_text(json.dumps({"schema": "bb-irivm-full-state-history-audit-v2", "status": "PASS_SCHEMA_AWARE_STORAGE_ONLY_CAPTURE_PENDING_EXACT_PREFIX_VERIFICATION", "label": self.evaluator.R4_LABEL, "additional_physical_calls": 0, "missing_counter_values_guessed_or_backfilled": False}), encoding="utf-8")
        closure_paths = {"registration": registration, "authorization": authorization, "reference_result": reference, "replay_result": replay, "replay_audit": replay_audit, "capture_manifest": capture_manifest, "capture_audit": capture_audit}
        bindings = {role: {"path": str(path.resolve()), "sha256": self.evaluator.ENGINE.sha256_file(path)} for role, path in closure_paths.items()}
        verification = directory / "verification.json"
        verification.write_text(json.dumps({
            "schema": "bb-irivm-selected-full-state-history-verification-v4",
            "status": "PASS_EXACT_FULL_STATE_HISTORY_OUTPUT_ONLY_REPLAY",
            "all_discrete_I_perp_values_exactly_reconstructed": True,
            "additional_physical_calls_from_capture": 0, "path_inventory": inventory, "bindings": bindings,
        }), encoding="utf-8")
        return verification, registration, authorization

    def test_freezer_to_real_shape_preflight_makes_zero_openmm_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            verification, registration, authorization = self.make_verification(directory)
            contract = directory / "contract.json"
            arguments = [
                "freeze_bb_dense_ivm_four_joint_v2_contract.py", "--output", str(contract),
                "--full-state-verification", str(verification), "--host-role", "remote",
                "--global-lock-path", str(directory / "a.lock"), "--peer-global-lock-path", str(directory / "b.lock"),
            ]
            old_r3 = directory / "old_r3_verification.json"
            old_payload = json.loads(verification.read_text(encoding="utf-8"))
            old_payload["schema"] = "bb-irivm-selected-full-state-history-verification-v3"
            old_r3.write_text(json.dumps(old_payload), encoding="utf-8")
            old_arguments = arguments.copy()
            old_arguments[old_arguments.index(str(verification))] = str(old_r3)
            with mock.patch.object(self.freezer, "CANONICAL_R4_REGISTRATION", registration), mock.patch.object(self.freezer, "CANONICAL_R4_AUTHORIZATION", authorization), mock.patch.object(sys, "argv", old_arguments), self.assertRaisesRegex(ValueError, "canonical PASS v4"):
                self.freezer.main()
            with mock.patch.object(self.freezer, "CANONICAL_R4_REGISTRATION", registration), mock.patch.object(self.freezer, "CANONICAL_R4_AUTHORIZATION", authorization), mock.patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
                self.freezer.main()
            env = {name: "1" for name in self.evaluator.ENGINE.THREAD_VARS}
            with mock.patch.object(self.evaluator, "CANONICAL_R4_REGISTRATION", registration), mock.patch.object(self.evaluator, "CANONICAL_R4_AUTHORIZATION", authorization), mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
                self.evaluator.ENGINE, "OpenMMBackend", side_effect=AssertionError("preflight instantiated OpenMM"),
            ), redirect_stdout(io.StringIO()) as captured:
                self.evaluator.main([
                    "--contract", str(contract), "--host-role", "remote",
                    "--output", str(directory / "unused.npz"), "--preflight-only",
                ])
            payload = json.loads(captured.getvalue())
            self.assertEqual(payload["status"], "ZERO_OPENMM_PREFLIGHT_PASS")
            self.assertEqual(payload["calls_per_image"], [2, 5, 10, 50])
            self.assertEqual(payload["optimization_N_F"], [60, 150, 300, 1329])
            self.assertEqual(payload["new_original_discrete_joint_EF_attempts_planned"], 0)
            self.assertEqual(payload["dense_joint_EF_attempts_planned"], 2000)
            self.assertEqual(payload["dense_full_force_vectors_planned_for_storage"], 2000)
            self.assertEqual(payload["optimization_N_F_contribution"], 0)
            first_source = Path(json.loads(verification.read_text(encoding="utf-8"))["path_inventory"][0]["shard"])
            with first_source.open("ab") as stream:
                stream.write(b"tamper")
            with mock.patch.object(self.evaluator, "CANONICAL_R4_REGISTRATION", registration), mock.patch.object(self.evaluator, "CANONICAL_R4_AUTHORIZATION", authorization), mock.patch.dict(os.environ, env, clear=False), self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.evaluator.main([
                    "--contract", str(contract), "--host-role", "remote",
                    "--output", str(directory / "unused.npz"), "--preflight-only",
                ])

    def test_fake_backend_stores_dense_full_force_and_closes_wal(self):
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
        ), mock.patch.multiple(evaluator.ENGINE, N_ATOMS=2, DIMENSION=6, N_DENSE=5):
            directory = Path(temporary)
            contract = directory / "contract.json"
            contract.write_text("{}\n", encoding="utf-8")
            topology = directory / "topology.pdb"
            topology.write_text("synthetic\n", encoding="utf-8")
            rows = []
            for history_index, (call, nf) in enumerate(zip(evaluator.CALLS, evaluator.EXPECTED_NF)):
                path = np.ascontiguousarray(
                    0.01 * np.arange(32, dtype=np.float64)[:, None] * np.arange(1, 7, dtype=np.float64)[None, :] + history_index
                )
                energy = np.ascontiguousarray(np.sum(path * path, axis=1))
                force = np.ascontiguousarray(-2.0 * path)
                source = directory / f"source_{call}.npz"
                np.savez(source, marker=np.asarray(call))
                rows.append({
                    "label": f"IVM_call{call}_NF{nf}", "optimization_N_F": nf,
                    "path_identifier": {"calls_per_image": call, "history_index": history_index},
                    "source_path": source.resolve(), "online_full_state_source": {"sha256": evaluator.ENGINE.sha256_file(source)},
                    "path": path, "path_sha256": evaluator.ENGINE.sha256_array(path),
                    "online_energy": energy, "online_energy_sha256": evaluator.ENGINE.sha256_array(energy),
                    "online_force": force, "online_force_sha256": evaluator.ENGINE.sha256_array(force),
                    "i_perp": float(evaluator.ENGINE.path_diagnostics(path, energy, force[1:-1])["i_perp"]),
                })
            output = directory / "ivm.npz"
            locks = (directory / "a.lock", directory / "b.lock")
            with mock.patch.object(evaluator.ENGINE, "OpenMMBackend", FakeBackend):
                evaluator.run(contract.resolve(), topology.resolve(), locks, rows, output)
            self.assertEqual(sum(instance.raw_joint_attempts for instance in FakeBackend.instances), 20)
            for call in evaluator.CALLS:
                with np.load(evaluator.shard_path_for(output, call), allow_pickle=False) as shard:
                    self.assertEqual(shard["dense_coordinates_A"].shape, (5, 6))
                    self.assertEqual(shard["dense_energy_eV"].shape, (5,))
                    self.assertEqual(shard["dense_force_eV_per_A"].shape, (5, 6))
                    self.assertTrue(np.array_equal(shard["dense_coordinates_A"][0], shard["selected_path"][0]))
                    self.assertTrue(np.array_equal(shard["dense_coordinates_A"][-1], shard["selected_path"][-1]))
                    self.assertEqual(int(shard["optimization_N_F_contribution"].item()), 0)
            audit = json.loads(output.with_suffix(".audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["new_original_discrete_joint_EF_attempts"], 0)
            self.assertEqual(audit["dense_WAL_shard_pointwise_closure"]["status"], "PASS")
            self.assertEqual(audit["dense_WAL_shard_pointwise_closure"]["lifetime"]["attempt_succeeded"], 20)
            with np.load(output, allow_pickle=False) as saved:
                tampered_arrays = {key: np.asarray(saved[key]).copy() for key in saved.files}
            tampered_arrays["energy_eV"][0, 0] += 1.0
            tampered = directory / "tampered.npz"
            np.savez(tampered, **tampered_arrays)
            shard_rows = [
                evaluator._validate_shard(
                    evaluator.shard_path_for(output, call), row,
                    evaluator.ENGINE.sha256_file(contract), evaluator.ENGINE.sha256_file(topology),
                    evaluator.ENGINE.sha256_file(Path(evaluator.__file__)), evaluator.ENGINE.sha256_file(evaluator.ENGINE_PATH),
                    evaluator.ENGINE.sha256_file(ROOT / "src" / "bb_core.py"),
                )
                for call, row in zip(evaluator.CALLS, rows)
            ]
            with self.assertRaisesRegex(ValueError, "energy_eV value mismatch"):
                evaluator._validate_aggregate(tampered, rows, shard_rows, evaluator.ENGINE.sha256_file(contract))
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                evaluator.run(contract.resolve(), topology.resolve(), locks, rows, output)

            output.with_suffix(".manifest.json").unlink()
            output.with_suffix(".audit.json").unlink()
            with mock.patch.object(evaluator.ENGINE, "OpenMMBackend", side_effect=AssertionError("finalization recovery must not call OpenMM")):
                evaluator.run(contract.resolve(), topology.resolve(), locks, rows, output)
            recovered = json.loads(output.with_suffix(".audit.json").read_text(encoding="utf-8"))
            self.assertEqual(recovered["complete_validated_shards_reused"], [2, 5, 10, 50])

    def test_failed_dense_attempt_is_not_reused_on_restart(self):
        evaluator = self.evaluator

        class FailingBackend:
            def __init__(self, *args, **kwargs):
                self.raw_joint_attempts = 0
                self.raw_joint_evaluations = 0

            def evaluate_joint(self, coordinate):
                self.raw_joint_attempts += 1
                if self.raw_joint_attempts == 3:
                    raise RuntimeError("synthetic dense failure")
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
        ), mock.patch.multiple(evaluator.ENGINE, N_ATOMS=2, DIMENSION=6, N_DENSE=5):
            directory = Path(temporary)
            contract = directory / "contract.json"
            contract.write_text("{}\n", encoding="utf-8")
            topology = directory / "topology.pdb"
            topology.write_text("synthetic\n", encoding="utf-8")
            rows = []
            for history_index, (call, nf) in enumerate(zip(evaluator.CALLS, evaluator.EXPECTED_NF)):
                path = np.ascontiguousarray(0.01 * np.arange(32)[:, None] * np.arange(1, 7)[None, :] + history_index, dtype=np.float64)
                energy = np.ascontiguousarray(np.sum(path, axis=1), dtype=np.float64)
                force = np.ascontiguousarray(-path, dtype=np.float64)
                source = directory / f"source_{call}.npz"
                np.savez(source, marker=np.asarray(call))
                rows.append({"label": f"IVM_call{call}_NF{nf}", "optimization_N_F": nf, "path_identifier": {"calls_per_image": call, "history_index": history_index}, "source_path": source.resolve(), "online_full_state_source": {"sha256": evaluator.ENGINE.sha256_file(source)}, "path": path, "path_sha256": evaluator.ENGINE.sha256_array(path), "online_energy": energy, "online_energy_sha256": evaluator.ENGINE.sha256_array(energy), "online_force": force, "online_force_sha256": evaluator.ENGINE.sha256_array(force), "i_perp": float(evaluator.ENGINE.path_diagnostics(path, energy, force[1:-1])["i_perp"])})
            output = directory / "retry.npz"
            locks = (directory / "a.lock", directory / "b.lock")
            with mock.patch.object(evaluator.ENGINE, "OpenMMBackend", FailingBackend), self.assertRaisesRegex(RuntimeError, "synthetic dense failure"):
                evaluator.run(contract.resolve(), topology.resolve(), locks, rows, output)
            self.assertFalse(evaluator.shard_path_for(output, 2).exists())
            with mock.patch.object(evaluator.ENGINE, "OpenMMBackend", GoodBackend):
                evaluator.run(contract.resolve(), topology.resolve(), locks, rows, output)
            audit = json.loads(output.with_suffix(".audit.json").read_text(encoding="utf-8"))
            lifetime = audit["dense_WAL_shard_pointwise_closure"]["lifetime"]
            self.assertEqual(lifetime["attempt_failed"], 1)
            self.assertEqual(lifetime["attempt_open"], 0)
            self.assertEqual(lifetime["retry_attempts"], 3)
            self.assertEqual(audit["individual_incomplete_WAL_attempts_reused"], 0)


if __name__ == "__main__":
    unittest.main()
