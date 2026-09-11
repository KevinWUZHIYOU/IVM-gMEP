#!/usr/bin/env python3
"""Zero-physical synthetic integration tests for R2 PDF/provenance closure."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


sys.dont_write_bytecode = True
matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


QA = load("bb_r2_pdf_qa_tested", HERE / "audit_final_pdfs_r2.py")
FINAL = load("bb_r2_delivery_tested", HERE / "finalize_r2_delivery.py")


class R2DeliveryTests(unittest.TestCase):
    def test_real_terminal_authorization_mode_normalized_binding(self):
        authorization = (
            FINAL.R2.CACHE_REPLICA_RUN.parent.parent / "authorizations" /
            f"{FINAL.R2.CACHE_REPLICA_RUN.name}.execution-authorization.json"
        )
        observed = FINAL.resolve_terminal_authorization({
            "authorization": {
                "path": str(authorization),
                "sha256": FINAL.sha256_file(authorization),
            }
        })
        self.assertEqual(observed["path"], str(authorization))
        self.assertEqual(observed["local_replica_mode_octal"], "0600")
        self.assertEqual(
            observed["remote_mode_verified_by_canonical_terminal_audit"], "0444"
        )
        self.assertFalse(observed["local_mode_was_changed_by_analysis"])

    def write_json(self, path: Path, value: dict) -> Path:
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        path.chmod(0o444)
        return path.resolve()

    def write_blob(self, path: Path, value: bytes = b"sealed fixture") -> Path:
        path.write_bytes(value); path.chmod(0o444); return path.resolve()

    def make_pdf(self, path: Path, role: str) -> Path:
        if role == "cost_to_precision":
            figure, axes = plt.subplots(1, 2, figsize=(7.0, 4.0))
            for axis in axes:
                axis.plot([1.0e8, 1.0e1], [0, 30000], label="IVM")
                axis.plot([1.0e8, 1.0e1], [0, 30000], label="ODE-NEB")
                axis.set_ylim(0.0, 30000.0)
                axis.set_yscale("linear")
            axes[0].set_xscale("log")
            axes[0].set_xlim(1.0e8, 1.0e1)
            axes[0].set_ylabel("Number of Forces")
            axes[1].set_xscale("linear")
            axes[1].set_xlim(5000.0, 0.0)
            axes[0].legend()
        elif role == "energy_vs_arc_length":
            figure, axis = plt.subplots(figsize=(7.0, 4.0))
            coordinate = np.linspace(0.0, 1.0, 32)
            axis.plot(coordinate, coordinate, label="IVM (NF 8982)")
            axis.plot(coordinate, 1.0 - coordinate, label="ODE-NEB (NF 30000)")
            axis.plot(
                coordinate, 0.6 - 0.2 * coordinate,
                label="Previous IVM (NF 9990)",
            )
            axis.plot(
                coordinate, 0.2 + 0.1 * coordinate,
                label="IVM trial (NF 14983; uncommitted)",
            )
            axis.set_xlabel("Normalized Cartesian arc length")
            axis.set_ylabel("Relative energy")
            axis.legend()
        else:
            figure, axis = plt.subplots(figsize=(7.0, 4.0))
            axis.plot([0, 1], [0, 1], label="IVM")
            axis.plot([0, 1], [1, 0], label="ODE-NEB")
            axis.set_title("NF evolution clipped")
            axis.legend()
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)
        path.chmod(0o444)
        return path.resolve()

    def test_full_synthetic_qa_collect_render_and_atomic_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = {
                key: root / expected.name
                for key, expected in FINAL.EXPECTED_OUTPUTS.items()
            }
            for role, key in QA.PDF_KEYS.items():
                self.make_pdf(outputs[key], role)
            source = root / "source"; source.mkdir()
            source_manifest = self.write_blob(source / "SOURCE_MANIFEST.json")
            run = root / "run"; run.mkdir()
            authorization = self.write_json(root / "authorization.json", {"status": "authorized"})
            terminal = self.write_json(root / "terminal.json", {
                "status": "PASS", "terminal_eligible": True,
                "formal_figure_eligible": True, "paid_batches": 1000,
                "paid_optimization_N_F": 30000,
                "physical_force_calls_by_external_audit": 0,
                "OpenMM_contexts_constructed_by_external_audit": 0,
                "authorization": {"path": str(authorization),
                                  "sha256": FINAL.sha256_file(authorization)},
            })
            cache = self.write_json(root / "cache.json", {
                "status": "FINAL_BOUND_TO_CANONICAL_TERMINAL_AUDIT",
                "processed_batches": 1000, "processed_optimization_N_F": 30000,
                "local_mutagen_replica_transport":
                    FINAL.R2.expected_cache_replica_transport(),
            })
            surface = self.write_blob(root / "surface.npz")
            surface_audit = self.write_json(root / "surface.audit.json", {"status": "PASS"})
            surface_adapter = self.write_json(root / "surface.adapter.json", {"status": "PASS"})
            ir_state = self.write_json(root / "ir-state.json", {"status": "PASS"})
            ir_verification = self.write_json(root / "ir-verification.json", {"status": "PASS"})
            registered_comparison = json.loads(
                (HERE / "ANALYSIS_INPUTS_R2.template.json").read_text()
            )["comparison"]
            manifest = {
                "schema": FINAL.R2.MANIFEST_SCHEMA,
                "status": "FINAL_INPUTS_FROZEN",
                "comparison": registered_comparison,
                "outputs": {"final": {key: str(path) for key, path in outputs.items()}},
                "inputs": {
                    "ode_source_root": str(source),
                    "formal_ode_neb": {
                        "run_dir": str(run),
                        "terminal_audit": {"path": str(terminal),
                                           "sha256": FINAL.sha256_file(terminal)},
                        "incremental_cache": {"path": str(cache),
                                              "sha256": FINAL.sha256_file(cache)},
                    },
                    "ir_centered_surface_bundle": {"path": str(surface),
                                                   "sha256": FINAL.sha256_file(surface)},
                    "ir_centered_surface_audit": {"path": str(surface_audit),
                                                  "sha256": FINAL.sha256_file(surface_audit)},
                    "surface_center_state_audit": {"path": str(surface_adapter),
                                                   "sha256": FINAL.sha256_file(surface_adapter)},
                    "ir_ivm_full_state": {
                        "state_binding": {"path": str(ir_state),
                                          "sha256": FINAL.sha256_file(ir_state)},
                        "verification": {"path": str(ir_verification),
                                         "sha256": FINAL.sha256_file(ir_verification)},
                    },
                },
            }
            manifest_path = self.write_json(root / "manifest.json", manifest)
            base = {
                "i_perp": 2.0, "delta_e32_eV": 3.0,
                "delta_e32_kcal_per_mol": 69.1815,
                "path_sha256": "a" * 64, "energy_sha256": "b" * 64,
                "force_sha256": "c" * 64,
            }
            methods = {
                "IVM": [{**base, "method": "IVM", "optimization_N_F": 8982}],
                "Previous IVM": [{**base, "method": "Previous IVM",
                                   "optimization_N_F": 9990}],
                "ODE-NEB": [{**base, "method": "ODE-NEB", "optimization_N_F": 30000,
                             "i_perp": 3.0, "delta_e32_kcal_per_mol": 80.0}],
                FINAL.R2.IR_DIAGNOSTIC_KEY: [{
                    **base,
                    "method": FINAL.R2.IR_DIAGNOSTIC_KEY,
                    "optimization_N_F": 14983,
                    "state_role": "terminal uncommitted charged tail diagnostic",
                    "source": {
                        "formally_reportable": False,
                        "terminal_uncommitted_charged_tail": True,
                    },
                }],
            }
            values = {"latest_observed_N_F": 8982, "best_i_perp": 2.0,
                      "best_delta_e32_eV": 3.0,
                      "best_delta_e32_kcal_per_mol": 69.1815}
            pack = {
                "schema": FINAL.R2.PACK_SCHEMA,
                "status": "COMPLETE_ZERO_FORCE_DISCRETE32_NF30000_R2_ANALYSIS",
                "manifest": {"path": str(manifest_path),
                             "sha256": FINAL.sha256_file(manifest_path)},
                "metric_protocol": {"path_images": 32, "densification": False},
                "histories": {key: rows for key, rows in methods.items()
                              if key != FINAL.R2.IR_DIAGNOSTIC_KEY},
                "diagnostic_uncommitted_states": methods[FINAL.R2.IR_DIAGNOSTIC_KEY],
                "diagnostic_exclusion": {
                    "excluded_from_formal_IVM_history": True,
                    "excluded_from_cost_to_precision": True,
                    "excluded_from_same_N_F_envelopes": True,
                    "permitted_only_in_energy_profile_and_evolution_as_labeled_diagnostic": True,
                },
                "evolution_surface_envelope_registration": registered_comparison[
                    "evolution_surface_envelope_registration"
                ],
                "legacy_evolution_layout_reference": registered_comparison[
                    "legacy_evolution_layout_reference"
                ],
                "same_N_F_budget_envelopes": [{
                    "budget_N_F": 30000,
                    "methods": {"IVM": values, "Previous IVM": values,
                                "ODE-NEB": {**values, "latest_observed_N_F": 30000}},
                }],
            }
            comparison = self.write_json(outputs["comparison_pack_json"], pack)
            with outputs["same_N_F_table_csv"].open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "budget_N_F", "method", "latest_observed_N_F", "best_i_perp",
                    "best_delta_e32_eV", "best_delta_e32_kcal_per_mol",
                ])
                writer.writeheader()
                for method, row in pack["same_N_F_budget_envelopes"][0]["methods"].items():
                    writer.writerow({"budget_N_F": 30000, "method": method, **row})
            outputs["same_N_F_table_csv"].chmod(0o444)
            argv = [
                "audit_final_pdfs_r2.py", "--manifest", str(manifest_path),
                "--comparison", str(comparison), "--render-dir", str(root / "renders"),
                "--output", str(outputs["pdf_qa_json"]), "--visual-confirmed",
            ]
            with mock.patch.object(sys, "argv", argv):
                QA.main()
            self.assertEqual(outputs["pdf_qa_json"].stat().st_mode & 0o777, 0o444)
            qa_document = json.loads(outputs["pdf_qa_json"].read_text())
            self.assertTrue(qa_document["checks"][
                "cost_to_precision_formal_curves_are_only_IVM_and_ODE_NEB"
            ])
            self.assertTrue(qa_document["checks"][
                "cost_to_precision_excludes_previous_IVM_and_uncommitted_trial"
            ])
            self.assertTrue(qa_document["checks"][
                "energy_profile_includes_previous_IVM_NF9990"
            ])
            self.assertIn(
                "Previous IVM NF9990 baseline",
                qa_document["line_semantics"]["energy_vs_arc_length"],
            )
            self.assertEqual(qa_document["main_pdf"], QA.main_pdf_binding())
            bad_qa = dict(qa_document)
            bad_qa["checks"] = {}
            bad_qa_path = self.write_json(root / "bad-empty-checks-qa.json", bad_qa)
            with self.assertRaisesRegex(RuntimeError, "schema/bindings/checks differ"):
                FINAL.validate_qa(
                    bad_qa_path, manifest_path, comparison, outputs,
                )
            bad_main_pdf_qa = dict(qa_document)
            bad_main_pdf_qa["main_pdf"] = {
                **qa_document["main_pdf"], "sha256": "0" * 64,
            }
            bad_main_pdf_qa_path = self.write_json(
                root / "bad-main-pdf-binding-qa.json", bad_main_pdf_qa,
            )
            with self.assertRaisesRegex(RuntimeError, "schema/bindings/checks differ"):
                FINAL.validate_qa(
                    bad_main_pdf_qa_path, manifest_path, comparison, outputs,
                )
            temporary_outputs = dict(outputs)
            with mock.patch.object(FINAL, "EXPECTED_OUTPUTS", temporary_outputs), \
                    mock.patch.object(
                        FINAL.R2, "resolve_replica_terminal_audit_file",
                        return_value=terminal.resolve(),
                    ), \
                    mock.patch.object(FINAL.R2, "load_complete_data_r2",
                                      return_value=(methods, {
                                          "selected_spring_k_eV_per_A2": 0.1,
                                      })), \
                    mock.patch.object(FINAL.R2, "load_surface_r2",
                                      return_value={"surface_energy_eV": np.zeros((59, 28))}), \
                    mock.patch.object(
                        FINAL, "resolve_terminal_authorization",
                        return_value={**FINAL.binding(authorization),
                                      "local_replica_mode_octal": "0444",
                                      "remote_mode_verified_by_canonical_terminal_audit": "0444",
                                      "local_mode_was_changed_by_analysis": False},
                    ):
                evidence = FINAL.collect_evidence(
                    manifest_path, comparison, outputs["same_N_F_table_csv"],
                    outputs["pdf_qa_json"],
                )
            text = FINAL.render_provenance(
                HERE / "FINAL_PROVENANCE_R2.template.md", evidence,
                outputs["completion_audit_json"],
            )
            self.assertNotIn("PENDING", text)
            document = FINAL.completion_audit(
                evidence, outputs["provenance_md"], text.encode("utf-8"),
                outputs["completion_audit_json"],
            )
            audit_bytes = json.dumps(document, sort_keys=True).encode() + b"\n"
            FINAL.publish_pair(
                outputs["provenance_md"], outputs["completion_audit_json"],
                text.encode("utf-8"), audit_bytes,
            )
            self.assertEqual(outputs["provenance_md"].stat().st_mode & 0o777, 0o444)
            self.assertEqual(document["physical_calls"], 0)
            self.assertNotIn("openmm", sys.modules)

    def test_manifest_template_has_versioned_document_outputs(self):
        template = json.loads((HERE / "ANALYSIS_INPUTS_R2.template.json").read_text())
        self.assertEqual(
            template["comparison"]["evolution_same_N_F_budgets"],
            [2040, 15000, 30000],
        )
        registration = template["comparison"][
            "evolution_surface_envelope_registration"
        ]
        self.assertFalse(registration["path_clipping_permitted"])
        self.assertTrue(registration["pre_final_protocol_amendment"])
        self.assertTrue(registration["selection_not_based_on_performance"])
        self.assertEqual(registration["alternative_geometry_valid_budget_N_F"], 990)
        self.assertTrue(all(
            row["strict_envelope_pass"]
            for row in registration[
                "alternative_geometry_valid_budget_projection"
            ].values()
        ))
        self.assertFalse(
            registration["excluded_budget_projection"]["ODE-NEB"]
            ["strict_envelope_pass"]
        )
        self.assertEqual(
            registration["excluded_budget_projection"]["ODE-NEB"]
            ["normal_max_A"],
            0.7038122887583026,
        )
        outputs = template["outputs"]["final"]
        for key in ("pdf_qa_json", "provenance_md", "completion_audit_json"):
            self.assertIn("ode_nf30000_r2", outputs[key])


if __name__ == "__main__":
    unittest.main()
