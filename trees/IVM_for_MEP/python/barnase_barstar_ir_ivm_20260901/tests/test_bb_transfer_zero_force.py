from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

import bb_core
import bb_ffprecon
import frozen_discrete_b040
import run_ivm_v9_discrete_ffrbk_takeover_grid_shim as grid
import run_barnase_barstar_ir_ivm_batch as batch
import run_barnase_barstar_ir_ivm_member as member
import storage_policy


ROOT = Path(__file__).resolve().parents[1]
CHIG_FROZEN_SHA = "a42da93b10ab9e235770569159cfaf5bd36e819ed169ccb619729b5de6df9ed4"


class FakeBackend:
    def __init__(self):
        self.raw_joint_attempts = 0
        self.raw_joint_evaluations = 0
        self.label = "synthetic-zero-force"

    def evaluate_joint(self, position):
        self.raw_joint_attempts += 1
        position = np.asarray(position, dtype=float)
        self.raw_joint_evaluations += 1
        return float(np.dot(position, position)), -2.0 * position


class BBTransferContractTests(unittest.TestCase):
    def test_frozen_numerical_core_is_byte_identical_to_chignolin(self):
        path = Path(frozen_discrete_b040.__file__)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), CHIG_FROZEN_SHA)

    def test_system_shape_and_central_tangent_are_frozen(self):
        self.assertEqual(bb_core.N_IMAGES, 32)
        self.assertEqual(bb_core.N_INTERNAL, 30)
        path = np.zeros((32, 6), dtype=float)
        path[:, 0] = np.arange(32, dtype=float)
        tangent = bb_core.central_tangents(path)
        np.testing.assert_array_equal(tangent[:, 0], np.ones(30))
        np.testing.assert_array_equal(tangent[:, 1:], np.zeros((30, 5)))

    def test_input_pointer_hashes_and_shape_without_openmm(self):
        manifest = json.loads((ROOT / "inputs" / "INPUT_POINTERS.json").read_text())
        source = Path(manifest["canonical_source_root"]) / "inputs"
        for name in ("complex_fixed.pdb", "common_initial_path32.npz"):
            expected = manifest["files"][name]
            if not (source / name).is_file():
                self.skipTest("canonical source checkout is not mounted")
            self.assertEqual(hashlib.sha256((source / name).read_bytes()).hexdigest(), expected)
        cache = ROOT / "inputs" / "endpoint_joint_ef_cache.npz"
        self.assertEqual(
            hashlib.sha256(cache.read_bytes()).hexdigest(),
            manifest["files"][cache.name],
        )
        with np.load(source / "common_initial_path32.npz", allow_pickle=False) as data:
            path = np.ascontiguousarray(data["initial_path"], dtype=np.float64)
        self.assertEqual(list(path.shape), manifest["initial_path_array_shape"])
        self.assertEqual(hashlib.sha256(path.tobytes()).hexdigest(), manifest["initial_path_array_sha256"])

    def test_staged_protocol_binds_current_construction(self):
        protocol = json.loads(
            (ROOT / "registrations" / "STAGED_SEARCH_PROTOCOL.json").read_text()
        )
        bound = protocol["bound_construction"]
        for key in ("campaign_preregistration", "migration_contract", "input_pointers"):
            target = ROOT / bound[key]["path"]
            self.assertEqual(
                hashlib.sha256(target.read_bytes()).hexdigest(),
                bound[key]["sha256"],
            )

    def test_inherited_endpoints_preserve_values_and_remove_two_attempts(self):
        path = np.zeros((32, 6), dtype=np.float64)
        path[:, 0] = np.arange(32, dtype=np.float64)
        path[:, 1] = np.linspace(0.0, 0.5, 32)

        live_backend = FakeBackend()
        live = [bb_core.ImageEvaluator(live_backend, index) for index in range(32)]
        live_energy, live_force = bb_core.evaluate_path(path, live)

        backend = FakeBackend()
        evaluators = bb_core.make_evaluators(backend, count=32)
        evaluators[0].preload_inherited(
            path[0], live_energy[0], live_force[0]
        )
        evaluators[-1].preload_inherited(
            path[-1], live_energy[-1], live_force[-1]
        )
        energy, force = bb_core.evaluate_path(path, evaluators)
        np.testing.assert_array_equal(energy, live_energy)
        np.testing.assert_array_equal(force, live_force)
        np.testing.assert_array_equal(
            bb_core.central_tangents(path), bb_core.central_tangents(path)
        )
        live_diag = bb_core.path_diagnostics(path, live_energy, live_force[1:-1])
        cache_diag = bb_core.path_diagnostics(path, energy, force[1:-1])
        self.assertEqual(live_diag.keys(), cache_diag.keys())
        for key in live_diag:
            np.testing.assert_array_equal(live_diag[key], cache_diag[key])

        audit = bb_core.audit_counter_consistency(backend, evaluators)
        self.assertTrue(audit["consistent"])
        self.assertEqual(audit["optimization_internal_nf"], 30)
        self.assertEqual(audit["endpoint_joint_ef_cache_misses"], 0)
        self.assertEqual(audit["endpoint_getstate_attempts"], 0)
        self.assertEqual(audit["endpoint_inherited_joint_ef_cache_entries"], 2)
        self.assertEqual(audit["number_of_forces"], 30)
        self.assertEqual(live_backend.raw_joint_attempts, 32)
        backend.raw_joint_attempts += 1
        failed = bb_core.audit_counter_consistency(backend, evaluators)
        self.assertFalse(failed["consistent"])
        self.assertEqual(failed["number_of_forces"], 31)

    def test_endpoint_cache_is_authoritative_and_endpoint_miss_fails_closed(self):
        with np.load(
            ROOT / "inputs" / "common_initial_path32.npz", allow_pickle=False
        ) as data:
            path = np.ascontiguousarray(data["initial_path"], dtype=np.float64)
        backend = FakeBackend()
        evaluators = bb_core.make_evaluators(backend, count=32)
        cache_audit = bb_core.preload_frozen_endpoint_cache(
            ROOT / "inputs" / "endpoint_joint_ef_cache.npz",
            path,
            evaluators,
        )
        self.assertEqual(
            cache_audit["energy_sha256"],
            bb_core.EXPECTED_ENDPOINT_ENERGY_SHA256,
        )
        evaluators[0].query(path[0], "both")
        evaluators[-1].query(path[-1], "both")
        self.assertEqual(backend.raw_joint_attempts, 0)
        shifted = path[0].copy()
        shifted[0] += 1e-12
        with self.assertRaises(bb_core.EvaluationBudgetReached):
            evaluators[0].query(shifted, "both")
        self.assertEqual(backend.raw_joint_attempts, 0)
        drift_audit = bb_core.audit_counter_consistency(backend, evaluators)
        self.assertEqual(drift_audit["endpoint_budget_blocked_misses"], 1)
        self.assertFalse(drift_audit["consistent"])

    def test_complete_band_preflight_uses_thirty_internal_images(self):
        frozen = SimpleNamespace()

        def solve(*, max_new_calls):
            return max_new_calls

        frozen.solve_frozen_discrete_b040_band = solve
        budget = SimpleNamespace(
            ala_core_module=bb_core,
            budget=91,
            successful_internal_joint_evaluations=30,
            full_band_preflight_calls=0,
            full_band_preflight_crops=0,
            full_band_preflight_zero_step_stops=0,
            complete_band_budget_stop_count=0,
        )
        installed = grid.install_full_band_budget_preflight(frozen, budget)
        try:
            self.assertEqual(frozen.solve_frozen_discrete_b040_band(max_new_calls=5), 2)
        finally:
            installed.restore()

    def test_call_free_builder_install_does_not_touch_backend_ledger(self):
        class Cell:
            rank = 0

        class Template:
            def __init__(self):
                self.cell = Cell()
                self.pbc = None

            def __len__(self):
                return 3132

            def set_cell(self, values):
                self.cell.rank = 3
                self.cell.values = tuple(values)

            def set_pbc(self, value):
                self.pbc = bool(value)

        backend = FakeBackend()
        bank = SimpleNamespace(template=Template(), build=lambda path: path)
        installed = bb_ffprecon.install_call_free_fast_builder(bank, build_workers=1)
        self.assertIs(installed, bank)
        self.assertEqual(backend.raw_joint_attempts, 0)
        source = inspect.getsource(bb_ffprecon._call_free_fast_build)
        self.assertNotIn("getState", source)
        self.assertNotIn("evaluate_joint", source)
        self.assertNotIn("raw_joint_attempts +=", source)


class StorageEquivalenceTests(unittest.TestCase):
    def test_synthetic_short_prefix_full_vs_thin_is_online_identical(self):
        full = storage_policy.SyntheticPrefixRecorder(
            storage_policy.LEGACY, save_all_paths=True
        )
        thin = storage_policy.SyntheticPrefixRecorder(storage_policy.COMMITTED)
        committed_calls = {3, 8}
        for call in (1, 2, 3, 5, 8, 13):
            path = np.arange(32 * 9, dtype=float).reshape(32, 9) + call
            committed = call in committed_calls
            np.testing.assert_array_equal(
                full.observe(call, path, reparam_committed=committed),
                thin.observe(call, path, reparam_committed=committed),
            )
        full_final = full.finalize(13, full.current_path)
        thin_final = thin.finalize(13, thin.current_path)
        np.testing.assert_array_equal(full_final, thin_final)
        self.assertEqual(full.online_digest, thin.online_digest)
        self.assertEqual([call for call, _ in thin.stored], [1, 3, 8, 13])
        full_by_call = {call: path for call, path in full.stored}
        for call, path in thin.stored:
            np.testing.assert_array_equal(path, full_by_call[call])


class BatchLockTests(unittest.TestCase):
    def test_disk_free_thresholds_are_tier_locked(self):
        self.assertEqual(batch.required_disk_free_bytes("smoke"), 2 * 1024**3)
        self.assertEqual(batch.required_disk_free_bytes("screen"), 2 * 1024**3)
        self.assertEqual(batch.required_disk_free_bytes("production"), 8 * 1024**3)
        with self.assertRaises(ValueError):
            batch.required_disk_free_bytes("other")

    def test_dual_lock_is_nonblocking_and_releases_partial_acquisition(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "a.lock"
            second = Path(directory) / "b.lock"
            with batch.acquire_all_host_locks((first, second)) as held:
                expected = sorted((str(first.resolve()), str(second.resolve())))
                self.assertEqual(held, expected)
                with self.assertRaises(RuntimeError):
                    with batch.acquire_all_host_locks((first, second)):
                        pass
            with batch.acquire_all_host_locks((second, first)) as held_again:
                self.assertEqual(held_again, expected)

    def test_result_validation_rejects_any_endpoint_blocked_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.npz"
            np.savez(
                result,
                run_label=np.array("registered-member"),
                number_of_forces=np.array(30),
                optimization_internal_nf=np.array(30),
                total_internal_nf_budget=np.array(30),
                total_internal_nf_consumed=np.array(30),
                total_internal_nf_budget_exhausted=np.array(True),
            )
            audit = {
                "consistent": True,
                "backend_raw_joint_attempts": 30,
                "backend_raw_joint_evaluations": 30,
                "sum_image_joint_evaluations": 30,
                "optimization_internal_nf": 30,
                "endpoint_joint_ef_cache_misses": 0,
                "endpoint_inherited_joint_ef_cache_entries": 2,
                "per_image_inherited_cache_entries": [1] + [0] * 30 + [1],
                "endpoint_budget_blocked_misses": 0,
            }
            audit_file = result.with_suffix(".audit.json")
            audit_file.write_text(json.dumps(audit), encoding="utf-8")
            batch.validate_result(
                result,
                expected_label="registered-member",
                expected_budget=30,
            )
            audit["endpoint_budget_blocked_misses"] = 1
            audit_file.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                batch.validate_result(
                    result,
                    expected_label="registered-member",
                    expected_budget=30,
                )
            audit["endpoint_budget_blocked_misses"] = 0
            audit_file.write_text(json.dumps(audit), encoding="utf-8")
            np.savez(
                result,
                run_label=np.array("registered-member"),
                number_of_forces=np.array(29),
                optimization_internal_nf=np.array(29),
                total_internal_nf_budget=np.array(30),
                total_internal_nf_consumed=np.array(29),
                total_internal_nf_budget_exhausted=np.array(False),
            )
            with self.assertRaises(RuntimeError):
                batch.validate_result(
                    result,
                    expected_label="registered-member",
                    expected_budget=30,
                )


class ExecutionAuthorizationTests(unittest.TestCase):
    def test_per_member_authorization_is_exact_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registration = root / "registrations" / "smoke" / "member.json"
            campaign = root / "registrations" / "CAMPAIGN_PREREGISTRATION.json"
            staged_protocol = root / "registrations" / "STAGED_SEARCH_PROTOCOL.json"
            contract = root / "contracts" / "MIGRATION_CONTRACT.json"
            ff_parity_gate = (
                root / "reports" / "gates" / "FF_BUILDER_PARITY_ZERO_FORCE.json"
            )
            registration.parent.mkdir(parents=True)
            contract.parent.mkdir(parents=True)
            registration.write_text('{"tier": "smoke"}\n', encoding="utf-8")
            campaign.write_text("{}\n", encoding="utf-8")
            staged_protocol.write_text("{}\n", encoding="utf-8")
            contract.write_text("{}\n", encoding="utf-8")
            ff_parity_gate.parent.mkdir(parents=True)
            ff_parity_gate.write_text("{}\n", encoding="utf-8")
            label = "registered-member"
            with self.assertRaises(RuntimeError):
                member._require_execution_authorization(root, label, registration)
            authorization = (
                root / "registrations" / "execution_authorizations" / f"{label}.json"
            )
            authorization.parent.mkdir(parents=True)
            payload = {
                "enabled": True,
                "label": label,
                "member_registration_sha256": member.sha256(registration),
                "campaign_preregistration_sha256": member.sha256(campaign),
                "migration_contract_sha256": member.sha256(contract),
                "staged_search_protocol_sha256": member.sha256(staged_protocol),
                "ff_builder_parity_gate_sha256": member.sha256(ff_parity_gate),
            }
            authorization.write_text(json.dumps(payload), encoding="utf-8")
            member._require_execution_authorization(root, label, registration)
            payload["label"] = "wrong"
            authorization.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                member._require_execution_authorization(root, label, registration)
            payload["label"] = label
            registration.write_text('{"tier": "screen"}\n', encoding="utf-8")
            payload["member_registration_sha256"] = member.sha256(registration)
            authorization.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                member._require_execution_authorization(root, label, registration)
            smoke_gate = root / "reports" / "gates" / "SMOKE_240_GATE.json"
            smoke_gate.write_text('{"status": "PASS"}\n', encoding="utf-8")
            payload["smoke_240_gate_sha256"] = member.sha256(smoke_gate)
            authorization.write_text(json.dumps(payload), encoding="utf-8")
            member._require_execution_authorization(root, label, registration)


if __name__ == "__main__":
    unittest.main()
