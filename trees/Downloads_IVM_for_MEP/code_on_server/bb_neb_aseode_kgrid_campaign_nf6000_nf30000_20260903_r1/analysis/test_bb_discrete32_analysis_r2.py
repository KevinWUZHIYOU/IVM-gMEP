#!/usr/bin/env python3
"""Zero-physical regression tests for the NF30000 R2 analysis adapters."""

from __future__ import annotations

import importlib
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import matplotlib
import numpy as np


sys.dont_write_bytecode = True
matplotlib.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bb-r2-analysis-test-mpl")

HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


R2 = load_module("bb_discrete32_analysis_r2_tested", HERE / "bb_discrete32_analysis_r2.py")
CACHE = load_module(
    "bb_discrete32_cache_r2_tested", HERE / "cache_formal_discrete32_incrementally_r2.py"
)
FINALIZER = load_module(
    "bb_discrete32_finalizer_r2_tested", HERE / "finalize_analysis_inputs_r2.py"
)


class R2AnalysisTests(unittest.TestCase):
    def test_terminal_replica_resolver_rejects_every_nonexact_path_first(self):
        with self.assertRaisesRegex(
                R2.BASE_MODULE.AnalysisError, "outside the exact Mutagen"):
            R2.resolve_replica_terminal_audit_file(
                Path("/private/tmp/not-the-registered-terminal-audit.json")
            )

    def test_r2_atomic_publisher_freezes_every_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "machine-readable.txt"

            def writer(path):
                path.write_text("sealed\n", encoding="utf-8")

            R2.publish_artifact_set_new_r2([(target, writer)])
            self.assertEqual(target.read_text(encoding="utf-8"), "sealed\n")
            self.assertEqual(target.stat().st_mode & 0o777, 0o444)
            with self.assertRaisesRegex(
                    R2.BASE_MODULE.AnalysisError, "refusing to overwrite"):
                R2.publish_artifact_set_new_r2([(target, writer)])

    def test_nf30000_cache_adapter_changes_only_budget_configuration(self):
        self.assertEqual(CACHE.TARGET_BATCHES, 1000)
        self.assertEqual(CACHE.TARGET_NF, 30000)
        self.assertEqual(CACHE.TARGET_STATE_NF, (240, 2040, 15000, 30000))
        self.assertEqual(CACHE.sha256_file(CACHE.BASE), CACHE.BASE_SHA256)

    def test_mutagen_transport_mode_exception_is_exact_and_fail_closed(self):
        receipt = CACHE._read_transport_receipt()
        self.assertEqual(
            CACHE.sha256_file(CACHE.TRANSPORT_RECEIPT),
            CACHE.TRANSPORT_RECEIPT_SHA256,
        )
        self.assertEqual(
            CACHE.TRANSPORT_RECEIPT.stat().st_mode & 0o777,
            0o444,
        )
        self.assertEqual(len(receipt["preaudit_diagnostic_local_caches"]), 2)
        self.assertTrue(all(
            not row["formal_input_eligible"]
            for row in receipt["preaudit_diagnostic_local_caches"]
        ))
        prestart = CACHE.MUTAGEN_MIRROR_RUN / "PRESTART.json"
        CACHE.mutagen_transport_readonly_regular(prestart, "formal PRESTART")
        self.assertTrue(CACHE._registered_writable_transport_role(
            CACHE.MUTAGEN_ARCHIVE / "batch_0333_nf09990.npz",
            "published shard 333",
        ))
        self.assertTrue(CACHE._registered_writable_transport_role(
            CACHE.MUTAGEN_ARCHIVE / "batch_0333_nf09990.npz",
            "full-state shard",
        ))
        self.assertFalse(CACHE._registered_writable_transport_role(
            CACHE.MUTAGEN_ARCHIVE / "batch_0333_nf09991.npz",
            "published shard 333",
        ))
        self.assertFalse(CACHE._registered_writable_transport_role(
            CACHE.MUTAGEN_ARCHIVE / "batch_333_nf09990.npz",
            "published shard 333",
        ))
        self.assertFalse(CACHE._registered_writable_transport_role(
            CACHE.MUTAGEN_ARCHIVE / "batch_1001_nf30030.npz",
            "published shard 1001",
        ))
        self.assertFalse(CACHE._registered_writable_transport_role(
            CACHE.MUTAGEN_TERMINAL_AUDIT,
            "not the canonical terminal role",
        ))
        with tempfile.TemporaryDirectory() as temporary:
            outside = Path(temporary) / "batch_0001_nf00030.npz"
            outside.write_bytes(b"not a registered transport replica")
            outside.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "outside the exact Mutagen"):
                CACHE.mutagen_transport_readonly_regular(outside, "untrusted shard")
        source_root = Path(json.loads(
            (HERE / "ANALYSIS_INPUTS_R2.template.json").read_text()
        )["inputs"]["ode_source_root"])
        recorder, _selection = CACHE.import_mutagen_transport_source(source_root)
        actual_shard = next(iter(sorted(
            (CACHE.MUTAGEN_MIRROR_RUN / "paid_batch_full_state").glob("batch_*.npz")
        )))
        recorder._regular_file(actual_shard, "full-state shard", readonly=True)
        exact_args = [
            "--source-root", str(CACHE.FROZEN_SOURCE_ROOT),
            "--run-dir", str(CACHE.MUTAGEN_MIRROR_RUN),
            "--output-dir", str(CACHE.FORMAL_OUTPUT_DIR),
        ]
        CACHE.validate_cli_scope(exact_args)
        with self.assertRaisesRegex(ValueError, "outside the exact registered"):
            CACHE.validate_cli_scope([
                "--source-root", str(CACHE.FROZEN_SOURCE_ROOT),
                "--run-dir", str(CACHE.MUTAGEN_MIRROR_RUN),
                "--output-dir", str(CACHE.FORMAL_OUTPUT_DIR.with_name("wrong")),
            ])
        for flag, wrong in (
            ("--source-root", CACHE.FROZEN_SOURCE_ROOT.with_name("wrong-source")),
            ("--run-dir", CACHE.MUTAGEN_MIRROR_RUN.with_name("wrong-run")),
        ):
            altered = list(exact_args)
            altered[altered.index(flag) + 1] = str(wrong)
            with self.assertRaisesRegex(ValueError, "outside the exact registered"):
                CACHE.validate_cli_scope(altered)
        CACHE.validate_cli_scope(exact_args + [
            "--terminal-audit", str(CACHE.MUTAGEN_TERMINAL_AUDIT),
        ])
        with self.assertRaisesRegex(ValueError, "not the exact canonical"):
            CACHE.validate_cli_scope(exact_args + [
                "--terminal-audit", str(CACHE.MUTAGEN_TERMINAL_AUDIT.with_name("wrong")),
            ])
        with tempfile.TemporaryDirectory() as temporary:
            bound = Path(temporary) / "cache.json"
            CACHE.write_bound_cache_json(bound, {"schema": CACHE.CACHE_SCHEMA})
            document = json.loads(bound.read_text(encoding="utf-8"))
            self.assertEqual(
                document["local_mutagen_replica_transport"],
                CACHE.transport_provenance(),
            )

    def test_mutagen_cache_rejects_unbound_previous_and_unterminated_wal_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unbound = root / "formal_discrete32_cache_through_batch_0001.json"
            unbound.write_text(json.dumps({
                "schema": CACHE.CACHE_SCHEMA,
                "processed_batches": 1,
            }) + "\n", encoding="utf-8")
            unbound.chmod(0o444)
            with self.assertRaisesRegex(ValueError, "lacks the exact current replica"):
                CACHE.latest_bound_cache(root)
            wal = root / "tail.jsonl"
            wal.write_bytes(b'{"event":"complete"}\n{"event":"mutable"}')
            self.assertEqual(
                CACHE.MODULE.complete_json_lines(wal),
                [{"event": "complete"}],
            )

    def test_nf30000_observer_flattens_one_published_full_state_batch(self):
        manifest = json.loads((HERE / "ANALYSIS_INPUTS_R2.template.json").read_text())
        source = Path(manifest["inputs"]["ode_source_root"])
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        recorder_module = importlib.import_module("src.full_batch_recorder")
        selection_module = importlib.import_module("src.kgrid_selection")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_label = "fresh-nonresume-bb-neb-aseode-k0p1-formal-nf30000-r2-test"
            run = root / run_label
            run.mkdir()
            endpoint = source / "inputs" / "endpoint_joint_ef_cache.npz"
            endpoints, endpoint_energy, endpoint_force = recorder_module._load_endpoint_cache(
                endpoint
            )
            endpoints = endpoints.reshape((2, recorder_module.ATOM_COUNT, 3))
            prestart = run / "PRESTART.json"
            prestart.write_text(json.dumps({"candidate": {
                "phase": "formal",
                "paid_batches": 1000,
                "terminal_optimization_N_F": 30000,
                "spring_k_eV_per_A2": 0.1,
                "run_label": run_label,
            }}) + "\n")
            os.chmod(prestart, 0o444)
            recorder = recorder_module.FullBatchRecorder(
                run,
                endpoint,
                spring_k=0.1,
                terminal_batches=1000,
                run_label=run_label,
                authorization_sha256="a" * 64,
                source_closure_sha256="b" * 64,
            )
            path = np.linspace(endpoints[0], endpoints[1], 32)
            path[1:-1, 0, 0] += 1.0e-5
            energy30 = np.linspace(-3.0, -2.0, 30)
            force30 = np.zeros((30, recorder_module.ATOM_COUNT, 3))
            force30[:, 0, 1] = 1.0e-3
            archive = recorder.record(
                batch_id=1,
                nf=30,
                role="current_initial",
                disposition="current_initial",
                paid_path32_A=path,
                energies30_eV=energy30,
                forces30_eV_per_A=force30,
                current_path32_A=path,
                current_state_sha256=recorder_module.array_sha256(path),
            )
            published_path = run / archive["relative_path"]
            unpublished = published_path.with_name("batch_0002_nf00060.npz")
            unpublished.write_bytes(published_path.read_bytes())
            unpublished.chmod(0o444)
            event = {
                "schema": "bb-aseode30-paid-wal-v1",
                "sequence": 0,
                "utc": "2026-09-03T00:00:00+00:00",
                "event": "batch_commit",
                "run_label": run_label,
                "method": "neb_ase_ode",
                "previous_event_sha256": None,
                "batch_id": 1,
                "nf_after": 30,
                "paid_state_role": "current_initial",
                "disposition": "current_initial",
                "paid_path32_sha256": recorder_module.array_sha256(path),
                "current_path32_sha256": recorder_module.array_sha256(path),
                "optimizer_metadata": {"full_state_archive": archive},
            }
            event["event_sha256"] = hashlib.sha256(
                CACHE.MODULE.canonical_json_bytes(event)
            ).hexdigest()
            optimization = run / "optimization_force_wal.jsonl"
            optimization.write_bytes(CACHE.MODULE.canonical_json_bytes(event) + b"\n")
            output = root / "cache"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "cache-r2",
                    "--source-root",
                    str(source),
                    "--run-dir",
                    str(run),
                    "--output-dir",
                    str(output),
                ]
                CACHE.MODULE.readonly_regular = CACHE._BASE_READONLY_REGULAR
                CACHE.MODULE.import_frozen_source = CACHE._BASE_IMPORT_FROZEN_SOURCE
                CACHE.MODULE.latest_cache = CACHE._BASE_LATEST_CACHE
                CACHE.MODULE.write_json_new = CACHE._BASE_WRITE_JSON_NEW
                CACHE.MODULE.main()
            finally:
                sys.argv = old_argv
            document = json.loads(
                (output / "formal_discrete32_cache_through_batch_0001.json").read_text()
            )
            self.assertEqual(document["target_batches"], 1000)
            self.assertEqual(document["target_optimization_N_F"], 30000)
            self.assertEqual(document["processed_batches"], 1)
            full_energy = np.concatenate(
                (np.asarray(endpoint_energy[:1]), energy30, np.asarray(endpoint_energy[1:]))
            )
            full_force = np.concatenate(
                (
                    np.asarray(endpoint_force[:1]).reshape(1, -1),
                    force30.reshape(30, -1),
                    np.asarray(endpoint_force[1:]).reshape(1, -1),
                ),
                axis=0,
            )
            expected_i, expected_e = selection_module._metric_pair(
                path.reshape(32, -1), full_energy, full_force
            )
            self.assertEqual(document["rows"][0]["i_perp"], expected_i)
            self.assertEqual(document["rows"][0]["delta_e32_eV"], expected_e)

    def test_exact_adopted_prior_selection_recomputes_without_openmm(self):
        manifest = json.loads((HERE / "ANALYSIS_INPUTS_R2.template.json").read_text())
        source = Path(manifest["inputs"]["ode_source_root"])
        R2.BASE_MODULE.import_ode_source(source)
        authorization = importlib.import_module("src.kgrid_execution_authorization")
        receipt, evidence = R2.load_adopted_selection(manifest, authorization)
        self.assertEqual(receipt["selected_spring_k_eV_per_A2"], 0.1)
        self.assertEqual(
            evidence["verification_mode"],
            "exact-prior-source-full-portable-recomputation",
        )
        self.assertNotIn("openmm", sys.modules)
        self.assertNotIn("simtk.openmm", sys.modules)

    def test_call318_remains_formal_and_call515_is_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = np.zeros((32, R2.BASE_MODULE.N_DOF), dtype=np.float64)
            path[:, 0] = np.arange(32, dtype=np.float64)
            energy = np.linspace(-2.0, 1.0, 32)
            force = np.zeros_like(path)
            force[1:-1, 1] = 0.25
            stored_i, _ = R2.BASE_MODULE.metric_pair(path, energy, force)

            committed = root / "call_0318_stable_last.npz"
            np.savez_compressed(
                committed,
                path32_A=path,
                energy_eV=energy,
                force_eV_per_A=force,
                optimization_N_F=np.asarray(8982),
                discrete_I_perp=np.asarray(stored_i),
            )
            terminal = root / "call_0515_stable_last.npz"
            np.savez_compressed(
                terminal,
                call_per_image=np.asarray(515),
                optimization_N_F=np.asarray(14983),
                path32_A=path,
                energy_eV=energy,
                force_eV_per_A=force,
                discrete_I_perp=np.asarray(stored_i),
                fully_evaluated_current_path=np.asarray(True),
                formal_reportability_pending_final_result=np.asarray(True),
                additional_physical_calls=np.asarray(0),
            )
            inventory = [
                {
                    "call_per_image": 318,
                    "optimization_N_F": 8982,
                    "shard": f"/remote/{committed.name}",
                    "shard_sha256": R2.BASE_MODULE.sha256_file(committed),
                    "path_sha256": R2.BASE_MODULE.sha256_array(path),
                    "energy_sha256": R2.BASE_MODULE.sha256_array(energy),
                    "force_sha256": R2.BASE_MODULE.sha256_array(force),
                    "formally_reportable": True,
                    "terminal_uncommitted_charged_tail": False,
                },
                {
                    "call_per_image": 515,
                    "optimization_N_F": 14983,
                    "shard": f"/remote/{terminal.name}",
                    "shard_sha256": R2.BASE_MODULE.sha256_file(terminal),
                    "path_sha256": R2.BASE_MODULE.sha256_array(path),
                    "energy_sha256": R2.BASE_MODULE.sha256_array(energy),
                    "force_sha256": R2.BASE_MODULE.sha256_array(force),
                    "formally_reportable": False,
                    "terminal_uncommitted_charged_tail": True,
                },
            ]
            verification = root / "verification.json"
            verification.write_text(json.dumps({
                "schema": "bb-irivm-selected-full-state-history-verification-v5b",
                "status": "PASS_TEST_FIXTURE",
                "additional_physical_calls_from_capture": 0,
                "optimization_N_F_contribution_from_capture": 0,
                "last_complete_call": 318,
                "actual_canonical_call_count": 515,
                "terminal_optimization_N_F": 14983,
                "path_inventory": inventory,
            }) + "\n")
            state_binding = root / "state_binding.json"
            state_binding.write_text(json.dumps({
                "schema": "bb-irivm-r2-formal-and-diagnostic-state-binding-v1",
                "status": "PASS_ZERO_PHYSICAL_INPUT_BINDING",
                "physical_force_calls": 0,
                "verification": {
                    "path": str(verification.resolve()),
                    "sha256": R2.BASE_MODULE.sha256_file(verification),
                },
                "formal_terminal": {
                    "call_per_image": 318,
                    "optimization_N_F": 8982,
                    "formally_reportable": True,
                    "path_sha256": R2.BASE_MODULE.sha256_array(path),
                    "energy_sha256": R2.BASE_MODULE.sha256_array(energy),
                    "force_sha256": R2.BASE_MODULE.sha256_array(force),
                    "shard": {
                        "path": f"/remote/{committed.name}",
                        "sha256": R2.BASE_MODULE.sha256_file(committed),
                    },
                },
                "diagnostic_returned_terminal": {
                    "call_per_image": 515,
                    "optimization_N_F": 14983,
                    "formally_reportable": False,
                    "state_role": "terminal_uncommitted_charged_tail",
                    "path_sha256": R2.BASE_MODULE.sha256_array(path),
                    "energy_sha256": R2.BASE_MODULE.sha256_array(energy),
                    "force_sha256": R2.BASE_MODULE.sha256_array(force),
                    "i_perp": stored_i,
                    "delta_e32_eV": 3.0,
                    "delta_e32_kcal_per_mol": 3.0 * R2.BASE_MODULE.EV_TO_KCAL,
                    "shard": {
                        "path": f"/remote/{terminal.name}",
                        "sha256": R2.BASE_MODULE.sha256_file(terminal),
                    },
                },
            }) + "\n")
            manifest = {
                "inputs": {
                    "ir_ivm_full_state": {
                        "verification": {
                            "path": str(verification),
                            "sha256": R2.BASE_MODULE.sha256_file(verification),
                        },
                        "shard_root_override": str(root),
                        "state_binding": {
                            "path": str(state_binding),
                            "sha256": R2.BASE_MODULE.sha256_file(state_binding),
                        },
                        "state_interpretation": dict(R2.IR_INTERPRETATION),
                    }
                }
            }
            official, diagnostic = R2.load_ir_ivm_r2(manifest, (30000,))
            self.assertEqual([row["optimization_N_F"] for row in official], [8982])
            self.assertEqual(diagnostic[0]["optimization_N_F"], 14983)
            self.assertFalse(diagnostic[0]["source"]["formally_reportable"])
            self.assertIn("uncommitted", diagnostic[0]["state_role"])

    def test_plots_keep_diagnostic_out_of_cost_curve_and_label_energy_line(self):
        path = np.zeros((32, R2.BASE_MODULE.N_DOF), dtype=np.float64)
        path[:, 0] = np.arange(32, dtype=np.float64)
        energy = np.linspace(-2.0, 1.0, 32)
        force = np.zeros_like(path)
        force[1:-1, 1] = 0.25
        formal = R2.BASE_MODULE.record(
            "IVM", 8982, path, energy, force, {}, state_role="formal"
        )
        ode = {**formal, "method": "ODE-NEB", "optimization_N_F": 30000}
        previous = {**formal, "method": "Previous IVM", "optimization_N_F": 9990}
        diagnostic = {
            **formal,
            "method": R2.IR_DIAGNOSTIC_KEY,
            "optimization_N_F": 14983,
            "state_role": "uncommitted diagnostic",
        }
        methods = {
            "IVM": [formal],
            "ODE-NEB": [ode],
            "Previous IVM": [previous],
            R2.IR_DIAGNOSTIC_KEY: [diagnostic],
        }
        cost = R2.plot_cost_r2(methods, live=False)
        self.assertEqual(len(cost.axes), 2)
        self.assertEqual(cost.axes[0].get_xscale(), "log")
        self.assertEqual(cost.axes[0].get_xlim(), (1.0e8, 1.0e1))
        self.assertEqual(cost.axes[1].get_xscale(), "linear")
        self.assertEqual(cost.axes[1].get_xlim(), (5000.0, 0.0))
        for axis in cost.axes:
            self.assertEqual(axis.get_yscale(), "linear")
            self.assertEqual(axis.get_ylim(), (0.0, 30000.0))
        self.assertEqual(
            [text.get_text() for text in cost.legends[0].get_texts()],
            ["IVM", "ODE-NEB"],
        )
        cost.axes[0].set_xlim(1.0e7, 1.0e1)
        with mock.patch.object(R2, "_BASE_PLOT_COST", return_value=cost):
            with self.assertRaisesRegex(
                    R2.BASE_MODULE.AnalysisError, "cost panel contract differs"):
                R2.plot_cost_r2(methods, live=False)
        cost.axes[0].set_xlim(1.0e8, 1.0e1)
        profile = R2.plot_energy_profiles_r2(methods, 30000)
        labels = [line.get_label() for line in profile.axes[0].lines]
        self.assertEqual(labels, [
            "IVM (NF 8982)",
            "ODE-NEB (NF 30000)",
            "Previous IVM (NF 9990)",
            "IVM trial (NF 14983; uncommitted)",
        ])
        self.assertTrue(all(len(line.get_xdata()) == 32 for line in profile.axes[0].lines))
        self.assertTrue(all(len(line.get_ydata()) == 32 for line in profile.axes[0].lines))
        R2.BASE_MODULE.plt.close(cost)
        R2.BASE_MODULE.plt.close(profile)

    def test_real_call515_arrays_and_metrics_are_hash_bound(self):
        manifest = json.loads((HERE / "ANALYSIS_INPUTS_R2.template.json").read_text())
        item = manifest["inputs"]["ir_ivm_full_state"]
        verification_path = R2.BASE_MODULE.resolve_binding(
            item["verification"], "real R5B verification"
        )
        verification = R2.BASE_MODULE.load_json(verification_path)
        row = R2._load_uncommitted_ir_terminal(
            verification_path, verification, item
        )
        self.assertEqual(row["optimization_N_F"], 14983)
        self.assertEqual(row["i_perp"], 15.513490059900775)
        self.assertEqual(row["delta_e32_eV"], 5.228353617533173)
        self.assertEqual(
            row["path_sha256"],
            "5a4ba3c5dc8f7063c1963f0bfd951f82356e60d0ad9c4fc17c0bc8b0d36d8569",
        )

    def test_real_r2b_surface_result_audit_and_adapter_load_zero_physical(self):
        adapter_path = Path(
            "/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/"
            "runs/diagnostics/bb_irivm_call515_nf14983_terminal_uncommitted_tail_"
            "0k_section_surface_center_adapter_20260903_r2b.audit.json"
        )
        adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
        manifest = json.loads((HERE / "ANALYSIS_INPUTS_R2.template.json").read_text())
        manifest["inputs"]["ir_centered_surface_bundle"] = {
            key: adapter["surface_result"][key] for key in ("path", "sha256")
        }
        manifest["inputs"]["ir_centered_surface_audit"] = {
            key: adapter["surface_audit"][key] for key in ("path", "sha256")
        }
        manifest["inputs"]["surface_center_state_audit"] = {
            "path": str(adapter_path),
            "sha256": R2.BASE_MODULE.sha256_file(adapter_path),
        }
        surface = R2.load_surface_r2(manifest)
        self.assertEqual(surface["surface_energy_eV"].shape, (59, 28))
        self.assertEqual(surface["center_path59_A"].shape, (59, 3132, 3))
        source_energy = surface["surface_energy_eV"].copy()
        display_surface, display_audit = R2._surface_for_evolution_display(surface)
        self.assertTrue(np.array_equal(surface["surface_energy_eV"], source_energy))
        display_relative = np.maximum(
            (display_surface["surface_energy_eV"]
             - display_surface["surface_energy_eV"][0, R2.SURFACE_ZERO_INDEX])
            * R2.BASE_MODULE.EV_TO_KCAL,
            0.0,
        )
        self.assertLessEqual(
            float(np.max(display_relative)),
            R2.EVOLUTION_DISPLAY_CAP_KCAL_PER_MOL + 1.0e-7,
        )
        self.assertGreater(display_audit["clipped_grid_values"], 0)
        self.assertGreater(
            display_audit["unclipped_maximum_kcal_per_mol"],
            R2.EVOLUTION_DISPLAY_CAP_KCAL_PER_MOL,
        )
        self.assertEqual(display_audit["physical_calls"], 0)
        endpoint_path = Path(manifest["inputs"]["endpoint_cache"]["path"])
        with np.load(endpoint_path, allow_pickle=False) as endpoint:
            self.assertEqual(endpoint["coordinates_A"].shape, (2, 9396))
            reshaped = np.asarray(endpoint["coordinates_A"], dtype=np.float64).reshape(
                2, 3132, 3
            )
        self.assertTrue(np.array_equal(
            surface["center_path59_A"][[0, -1]], reshaped
        ))
        self.assertNotIn("openmm", sys.modules)

    def test_real_surface_adapter_passes_finalizer_sized_binding_gate(self):
        adapter_path = Path(
            "/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/"
            "runs/diagnostics/bb_irivm_call515_nf14983_terminal_uncommitted_tail_"
            "0k_section_surface_center_adapter_20260903_r2b.audit.json"
        )
        adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
        manifest = json.loads((HERE / "ANALYSIS_INPUTS_R2.template.json").read_text())
        FINALIZER.validate_surface_adapter(
            adapter,
            manifest["inputs"]["surface_center_state"],
            Path(adapter["surface_result"]["path"]),
            Path(adapter["surface_audit"]["path"]),
        )
        self.assertEqual(set(adapter["surface_result"]), {"path", "sha256", "bytes"})
        self.assertEqual(set(adapter["surface_audit"]), {"path", "sha256", "bytes"})
        self.assertNotIn("openmm", sys.modules)

    def test_finalizer_dry_run_with_synthetic_ode_and_real_surface_chain(self):
        adapter_path = Path(
            "/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/"
            "runs/diagnostics/bb_irivm_call515_nf14983_terminal_uncommitted_tail_"
            "0k_section_surface_center_adapter_20260903_r2b.audit.json"
        )
        adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
        surface_path = Path(adapter["surface_result"]["path"])
        surface_audit_path = Path(adapter["surface_audit"]["path"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "synthetic-final-ode-run"
            run_dir.mkdir()
            terminal_path = root / "terminal-audit.json"
            terminal_path.write_text(json.dumps({
                "status": "PASS",
                "terminal_eligible": True,
                "formal_figure_eligible": True,
                "candidate": {
                    "phase": "formal",
                    "spring_k_eV_per_A2": 0.1,
                },
                "paid_batches": R2.FORMAL_BATCHES,
                "paid_optimization_N_F": R2.FORMAL_NF,
            }), encoding="utf-8")
            terminal_path.chmod(0o444)
            terminal_binding = {
                "path": str(terminal_path.resolve()),
                "sha256": R2.BASE_MODULE.sha256_file(terminal_path),
            }
            cache_path = root / "formal-cache.json"
            cache_path.write_text(json.dumps({
                "status": "FINAL_BOUND_TO_CANONICAL_TERMINAL_AUDIT",
                "partial_is_final": False,
                "target_batches": R2.FORMAL_BATCHES,
                "target_optimization_N_F": R2.FORMAL_NF,
                "processed_batches": R2.FORMAL_BATCHES,
                "processed_optimization_N_F": R2.FORMAL_NF,
                "canonical_terminal_audit": terminal_binding,
                "physical_calls_started": 0,
                "openmm_contexts_constructed": 0,
                "bb_locks_acquired": 0,
                "local_mutagen_replica_transport":
                    R2.expected_cache_replica_transport(),
            }), encoding="utf-8")
            cache_path.chmod(0o444)
            output = root / "ANALYSIS_INPUTS_R2.synthetic-final.json"
            argv = [
                "finalize_analysis_inputs_r2.py",
                "--template", str(HERE / "ANALYSIS_INPUTS_R2.template.json"),
                "--run-dir", str(run_dir),
                "--terminal-audit", str(terminal_path),
                "--incremental-cache", str(cache_path),
                "--surface-result", str(surface_path),
                "--surface-audit", str(surface_audit_path),
                "--surface-center-adapter", str(adapter_path),
                "--output", str(output),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                    FINALIZER.R2, "resolve_replica_terminal_audit_file",
                    return_value=terminal_path.resolve()):
                FINALIZER.main()
            finalized = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(finalized["status"], "FINAL_INPUTS_FROZEN")
            self.assertEqual(
                finalized["inputs"]["ir_centered_surface_bundle"],
                {"path": str(surface_path),
                 "sha256": R2.BASE_MODULE.sha256_file(surface_path)},
            )
            self.assertEqual(finalized["finalization"]["physical_force_calls"], 0)
            self.assertEqual(output.stat().st_mode & 0o777, 0o444)
            self.assertNotIn("openmm", sys.modules)

    def test_surface_wal_requires_every_fresh_coordinate_and_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "surface.npz"
            result.write_bytes(b"synthetic sealed result")
            wal = root / "surface.energy.wal.jsonl"
            energy = np.linspace(
                -1.0, 1.0, R2.SURFACE_STORED_VALUES, dtype=np.float64
            ).reshape(R2.SURFACE_POINTS, R2.SURFACE_NORMAL_POINTS)
            hashes = np.empty(
                (R2.SURFACE_POINTS, R2.SURFACE_NORMAL_POINTS), dtype="<U64"
            )
            for logical_index in range(R2.SURFACE_STORED_VALUES):
                i, j = divmod(logical_index, R2.SURFACE_NORMAL_POINTS)
                hashes[i, j] = hashlib.sha256(
                    f"coordinate-{logical_index}".encode()
                ).hexdigest()
            events = [{
                "schema": R2.SURFACE_WAL_SCHEMA,
                "status": "OPEN_APPEND_ONLY_FSYNC",
                "stage": "surface",
                "fresh_energy_only_attempts_planned": R2.SURFACE_FRESH_VALUES,
                "maximum_attempt_started_records": R2.SURFACE_FRESH_VALUES,
                "force_calls_planned": 0,
                "optimization_N_F_contribution": 0,
            }]
            excluded = {
                R2.SURFACE_ZERO_INDEX,
                (R2.SURFACE_POINTS - 1) * R2.SURFACE_NORMAL_POINTS
                + R2.SURFACE_ZERO_INDEX,
            }
            for logical_index in range(R2.SURFACE_STORED_VALUES):
                if logical_index in excluded:
                    continue
                i, j = divmod(logical_index, R2.SURFACE_NORMAL_POINTS)
                attempt = f"surface:{logical_index:04d}:attempt001"
                common = {
                    "attempt_id": attempt,
                    "stage": "surface",
                    "logical_index": logical_index,
                    "center_index": i,
                    "normal_index": j,
                    "coordinate_sha256": str(hashes[i, j]),
                }
                events.append({"event": "attempt_started", **common})
                events.append({
                    "event": "attempt_succeeded",
                    **common,
                    "energy_eV": float(energy[i, j]),
                    "energy_scalar_sha256": R2.BASE_MODULE.sha256_array(
                        np.asarray(energy[i, j], dtype=np.float64)
                    ),
                })
            events.append({
                "event": "result_published",
                "stage": "surface",
                "result": str(result.resolve()),
                "result_sha256": R2.BASE_MODULE.sha256_file(result),
                "result_bytes": result.stat().st_size,
                "result_schema": R2.SURFACE_RESULT_SCHEMA,
            })
            wal.write_text(
                "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
                encoding="utf-8",
            )
            audit = {
                "energy_WAL_lifetime": {
                    "attempt_started": R2.SURFACE_FRESH_VALUES,
                    "attempt_succeeded": R2.SURFACE_FRESH_VALUES,
                    "attempt_failed": 0,
                    "attempt_open": 0,
                    "accepted_unique_successes": R2.SURFACE_FRESH_VALUES,
                    "retry_attempts": 0,
                },
                "authorization_lifetime_attempt_started_across_both_stages":
                    R2.SURFACE_FRESH_VALUES,
            }
            R2._validate_surface_wal(wal, result.resolve(), hashes, energy, audit)

    def test_template_waits_for_ode_and_surface_without_rendering(self):
        manifest = json.loads((HERE / "ANALYSIS_INPUTS_R2.template.json").read_text())
        missing = R2.readiness_r2(manifest)
        self.assertIn("sealed formal_ode_neb run/terminal audit", missing)
        self.assertIn("final zero-physical NF30000 incremental cache", missing)
        self.assertIsNone(manifest["inputs"]["ir_centered_surface_bundle"])
        self.assertIsNone(manifest["inputs"]["ir_centered_surface_audit"])
        self.assertNotIn("openmm", sys.modules)


if __name__ == "__main__":
    unittest.main()
