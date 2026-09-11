"""Zero-physical tests for the R5B late-six dual-host launch closure."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def load(name: str, relative: str):
    path = ROOT / "scripts" / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


F = load(
    "bb_cost_late_six_freezer_r4_tested",
    "freeze_bb_irivm_cost_late_six_arc500_r4.py",
)
R = load(
    "bb_cost_late_six_runner_r4_tested",
    "run_bb_irivm_cost_late_six_arc500_r4.py",
)
A = load(
    "bb_cost_late_six_authorizer_r4_tested",
    "authorize_bb_irivm_cost_late_six_arc500_r4.py",
)


class LateSixLaunchR4Test(unittest.TestCase):
    def test_preparation_exact_split_and_zero_physical(self):
        with mock.patch.object(
            F.DELEGATE.ENGINE.OpenMMBackend,
            "__init__",
            side_effect=AssertionError("preparation constructed OpenMM"),
        ):
            payload = F.build_preparation(require_preselection_absence=False)
        self.assertFalse(payload["R5B_final_artifacts_read"])
        self.assertFalse(payload["selection_read"])
        self.assertFalse(payload["metric_values_read"])
        self.assertEqual(
            payload["host_plans"]["remote"]["selection_roles"],
            list(F.SELECTOR.REMOTE_ROLES),
        )
        self.assertEqual(
            payload["host_plans"]["local"]["selection_roles"],
            list(F.SELECTOR.LOCAL_ROLES),
        )
        self.assertEqual(payload["host_plans"]["remote"]["fresh_joint_EF_calls"], 1992)
        self.assertEqual(payload["host_plans"]["local"]["fresh_joint_EF_calls"], 996)
        self.assertEqual(
            payload["scientific_policy"]["fresh_dense_interior_joint_EF_calls_per_path"],
            498,
        )
        self.assertEqual(payload["scientific_policy"]["optimization_N_F_contribution"], 0)

    def test_preparation_does_not_read_selection_or_r5b_final_artifacts(self):
        original = F.SELECTOR.read_json
        seen = []

        def observed(path):
            seen.append(Path(path).resolve())
            return original(path)

        with mock.patch.object(F.SELECTOR, "read_json", side_effect=observed):
            F.build_preparation(require_preselection_absence=False)
        forbidden = {
            F.SELECTOR.CANONICAL_SELECTION,
            F.SELECTOR.CANONICAL_PRIMARY,
            F.SELECTOR.CANONICAL_SUPPLEMENTAL,
            F.SELECTOR.CANONICAL_MANIFEST,
            F.SELECTOR.CANONICAL_INDEX,
        }
        self.assertTrue(forbidden.isdisjoint(seen))

    def test_command_templates_are_preflight_then_authorized_run(self):
        payload = F.build_preparation(require_preselection_absence=False)
        commands = payload["command_templates"]
        self.assertEqual(commands["environment"], F.SELECTOR.THREAD_CAPS)
        self.assertEqual(
            commands["remote_ssh_argv_prefix"][-1], "wuzhiyou@100.74.85.3"
        )
        for role in ("remote", "local"):
            self.assertIn("--preflight-only", commands[f"{role}_preflight"])
            self.assertNotIn("--authorization", commands[f"{role}_preflight"])
            self.assertNotIn("--preflight-only", commands[f"{role}_run"])
            self.assertIn("--authorization", commands[f"{role}_run"])
            self.assertIn("--authorization-gate", commands[f"{role}_run"])
        self.assertLess(
            list(commands).index("publication_mutagen_gate"),
            list(commands).index("remote_preflight"),
        )
        self.assertLess(
            list(commands).index("authorization_mutagen_gate"),
            list(commands).index("remote_run"),
        )

    def test_mutagen_template_and_snapshot_are_exact(self):
        self.assertIn("\n", F.MUTAGEN_LIST_TEMPLATE)
        self.assertNotIn("\\n", F.MUTAGEN_LIST_TEMPLATE)
        completed = [
            mock.Mock(returncode=0, stdout="", stderr=""),
            mock.Mock(
                returncode=0,
                stdout=(
                    f"{F.SELECTOR.MUTAGEN['name']}|"
                    f"{F.SELECTOR.MUTAGEN['identifier']}|Watching|true|true\n"
                ),
                stderr="",
            ),
            mock.Mock(returncode=0, stdout="Status: Watching for changes\n", stderr=""),
        ]
        with mock.patch.object(F.subprocess, "run", side_effect=completed):
            snapshot = F._mutagen_snapshot()
        self.assertEqual(snapshot["flush_exit_code"], 0)
        self.assertTrue(snapshot["alpha_connected"] and snapshot["beta_connected"])
        self.assertEqual(snapshot["transition_problems"], 0)

    def test_source_row_reuses_full_r5b_state_without_backend(self):
        endpoint = F.DELEGATE._load_endpoint_cache()
        fraction = np.linspace(0.0, 1.0, F.N_IMAGES)[:, None]
        path = (
            (1.0 - fraction) * endpoint["coordinates"][0]
            + fraction * endpoint["coordinates"][1]
        )
        energy = np.zeros(F.N_IMAGES, dtype=np.float64)
        force = np.zeros_like(path)
        energy[[0, -1]] = endpoint["energies"]
        force[[0, -1]] = endpoint["forces"]
        counters = np.zeros(F.N_IMAGES, dtype=np.int64)
        counters[1] = 2000
        requests = np.ones(F.N_IMAGES, dtype=np.int64)
        diagnostic = F.DELEGATE.ENGINE.path_diagnostics(
            path, np.zeros(F.N_IMAGES), force[1:-1]
        )["i_perp"]
        with tempfile.TemporaryDirectory() as raw:
            shard = Path(raw) / "source.npz"
            np.savez(
                shard,
                call_per_image=np.asarray(70),
                optimization_N_F=np.asarray(2000),
                path32_A=path,
                energy_eV=energy,
                force_eV_per_A=force,
                path_sha256=np.asarray(F.DELEGATE.verified_array_sha256(path)),
                energy_sha256=np.asarray(F.DELEGATE.ENGINE.sha256_array(energy)),
                force_sha256=np.asarray(F.DELEGATE.ENGINE.sha256_array(force)),
                joint_evaluation_vector=counters,
                force_request_vector=requests,
                discrete_I_perp=np.asarray(diagnostic),
                fully_evaluated_current_path=np.asarray(True),
                additional_physical_calls=np.asarray(0),
            )
            selected = {
                "history_index": 69,
                "call_per_image": 70,
                "optimization_N_F": 2000,
                "selected_path_sha256": F.DELEGATE.verified_array_sha256(path),
                "source_full_state_shard": str(shard),
                "source_full_state_shard_sha256": F.SELECTOR.sha256_file(shard),
                "selection_role": "nearest_unused_to_NF2000",
            }
            with mock.patch.object(
                F.DELEGATE.ENGINE.OpenMMBackend,
                "__init__",
                side_effect=AssertionError("source loading constructed OpenMM"),
            ):
                frozen, runtime = F._source_row(selected, 0, "local")
        self.assertEqual(frozen["original32_new_physical_calls"], 0)
        self.assertEqual(frozen["fresh_dense_joint_EF_calls"], 498)
        self.assertEqual(frozen["discrete_I_perp"], diagnostic)
        self.assertEqual(runtime["path"].shape, (32, 9396))
        self.assertEqual(runtime["source_counter_vector"][1], 2000)

    def test_delegate_configuration_is_scoped_and_balanced(self):
        original = {
            key: getattr(R.DELEGATE, key)
            for key in (
                "N_PATHS",
                "TOTAL_STORED_JOINT_EF",
                "MAXIMUM_FRESH_ATTEMPT_STARTED",
                "SHARD_SCHEMA",
            )
        }
        state = {"rows": [{}, {}, {}, {}]}
        with R.configured_delegate(state):
            self.assertEqual(R.DELEGATE.N_PATHS, 4)
            self.assertEqual(R.DELEGATE.TOTAL_STORED_JOINT_EF, 2000)
            self.assertEqual(R.DELEGATE.MAXIMUM_FRESH_ATTEMPT_STARTED, 1992)
            self.assertEqual(R.DELEGATE.SHARD_SCHEMA, R.SHARD_SCHEMA)
        for key, value in original.items():
            self.assertEqual(getattr(R.DELEGATE, key), value)

    def test_preflight_receipt_never_constructs_backend(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            contract = directory / "contract.json"
            contract.write_text("{}\n", encoding="utf-8")
            state = {
                "host_role": "remote",
                "rows": [
                    {"selection_role": role}
                    for role in F.SELECTOR.REMOTE_ROLES
                ],
                "contract": {
                    "execution_authorization_output": str(directory / "auth.json"),
                    "preflight_receipt_output": str(directory / "preflight.json"),
                },
                "contract_path": contract,
                "result_output": (directory / "result.npz").resolve(),
            }
            physical = {
                "OpenMM_backend_constructed": False,
                "lock_probe": "PASS_NONBLOCKING",
                "fresh_joint_EF_attempts_planned": 1992,
                "original_discrete_new_calls": 0,
                "optimization_N_F_contribution": 0,
                "remote_host_probe": {
                    "hostname": "Mac-mini.local",
                    "platform": "Darwin",
                    "machine": "arm64",
                },
                "lifetime": {
                    "attempt_started": 0,
                    "attempt_succeeded": 0,
                    "attempt_failed": 0,
                    "attempt_open": 0,
                    "retry_attempts": 0,
                },
            }
            with mock.patch.object(R.DELEGATE, "preflight", return_value=physical), mock.patch.object(
                R.DELEGATE.ENGINE.OpenMMBackend,
                "__init__",
                side_effect=AssertionError("preflight constructed OpenMM"),
            ), mock.patch.object(R, "bind", return_value={"path": "synthetic", "sha256": "0" * 64}):
                receipt = R.build_preflight_receipt(state, state["result_output"])
        self.assertEqual(receipt["status"], F.PREFLIGHT_STATUS)
        self.assertFalse(receipt["OpenMM_imported_or_called"])
        self.assertEqual(receipt["fresh_joint_EF_attempts_planned"], 1992)
        self.assertEqual(receipt["optimization_N_F_contribution"], 0)

    def test_authorizer_fails_closed_on_nonzero_wal_lifetime(self):
        state = {
            "host_role": "local",
            "rows": [{"selection_role": role} for role in F.SELECTOR.LOCAL_ROLES],
            "contract": {
                "preflight_receipt_output": "/synthetic/preflight.json"
            },
            "contract_path": Path("/synthetic/contract.json"),
            "result_output": Path("/synthetic/result.npz"),
        }
        payload = {
            "schema": A.FREEZER.PREFLIGHT_SCHEMA,
            "status": A.FREEZER.PREFLIGHT_STATUS,
            "host_role": "local",
            "selected_path_count": 2,
            "selection_roles": list(F.SELECTOR.LOCAL_ROLES),
            "fresh_joint_EF_attempts_planned": 996,
            "stored_arc500_joint_EF_values_planned": 1000,
            "inherited_endpoint_joint_EF_values_planned": 4,
            "original32_joint_EF_reused_per_path": 32,
            "original32_new_physical_calls": 0,
            "optimization_N_F_contribution": 0,
            "OpenMM_imported_or_called": False,
            "result_output": "/synthetic/result.npz",
            "bindings": {
                "preparation": {"path": "x", "sha256": "0" * 64},
                "bundle": {"path": "x", "sha256": "0" * 64},
                "publication_Mutagen_gate": {"path": "x", "sha256": "0" * 64},
                "contract": {"path": "x", "sha256": "0" * 64},
                "authorized_runner": {"path": "x", "sha256": "0" * 64},
            },
            "physical_preflight": {
                "lock_probe": "PASS_NONBLOCKING",
                "OpenMM_backend_constructed": False,
                "fresh_joint_EF_attempts_planned": 996,
                "lifetime": {
                    "attempt_started": 1,
                    "attempt_failed": 0,
                    "attempt_open": 0,
                    "retry_attempts": 0,
                },
            },
        }
        with tempfile.TemporaryDirectory() as raw:
            preflight = Path(raw) / "preflight.json"
            preflight.write_text(json.dumps(payload), encoding="utf-8")
            state["contract"]["preflight_receipt_output"] = str(preflight)
            with mock.patch.object(A.FREEZER, "bind", return_value={"path": "x", "sha256": "0" * 64}):
                with self.assertRaisesRegex(ValueError, "lifetime gate"):
                    A.validate_preflight(state, preflight)

    def test_noncanonical_selection_fails_before_reading(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "selection.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "noncanonical R3 selection"):
                F._selection_closure(path)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "frozen.json"
            F.write_exclusive(path, {"first": True})
            with self.assertRaises(FileExistsError):
                F.write_exclusive(path, {"second": True})

    def test_frozen_preparation_validates_when_present(self):
        if not F.PREPARATION.exists():
            self.skipTest("canonical preparation is not frozen yet")
        payload = F.validate_preparation(F.PREPARATION)
        self.assertEqual(payload["status"], F.PREPARATION_STATUS)


if __name__ == "__main__":
    unittest.main()
