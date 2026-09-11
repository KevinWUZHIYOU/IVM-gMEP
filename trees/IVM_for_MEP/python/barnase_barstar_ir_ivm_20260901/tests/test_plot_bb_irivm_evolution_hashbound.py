from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import plot_bb_irivm_evolution_hashbound as plotter  # noqa: E402


REFERENCE_4080 = ROOT / (
    "runs/screen/"
    "ivm_bb_irivm_a1e0_r01_m05_inf4080_r1_central_state_ivm_"
    "cfg47c24114c93c_budget4080_storagecommitted.npz"
)
EVOLUTION_TEMPLATE = (
    ROOT
    / "registrations"
    / "BB_IRIVM_EVOLUTION_FIGURE_CONTRACT_TEMPLATE_20260902_R1.json"
)


def scalar(value):
    return np.asarray(value)


class EvolutionFigureContractTests(unittest.TestCase):
    def test_inert_template_is_rejected_before_any_selected_npz_load(self) -> None:
        contract = json.loads(EVOLUTION_TEMPLATE.read_text(encoding="utf-8"))
        with mock.patch.object(plotter.np, "load", side_effect=AssertionError("np.load called")):
            with self.assertRaisesRegex(RuntimeError, "not a final frozen-input contract"):
                plotter.validate_json_gates_before_selected_npz(
                    EVOLUTION_TEMPLATE, contract
                )

    def test_two_selected_path_hash_conventions_are_kept_distinct(self) -> None:
        with np.load(REFERENCE_4080, allow_pickle=False) as saved:
            calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
            index = int(np.flatnonzero(calls == 7)[0])
            path = np.asarray(saved["stored_paths"][index], dtype=np.float64)
        self.assertEqual(
            plotter.verifier_dtype_shape_sha256(path),
            "ea3cd456fca7103d67d227bb095b93e2d7fba3c794e6081b7ac1951d53de9b7f",
        )
        self.assertEqual(
            plotter.canonical_le_float64_sha256(path),
            "d737fa00edc9f2e371f2a92e24cfe946c50a1a70d87205bddd801716af2573b3",
        )
        self.assertNotEqual(
            plotter.verifier_dtype_shape_sha256(path),
            plotter.canonical_le_float64_sha256(path),
        )

    def test_uncommitted_terminal_tail_is_not_an_eligible_panel_path(self) -> None:
        stored_paths = np.arange(5 * 32 * 6, dtype=np.float64).reshape(5, 32, 6)
        initial = stored_paths[0].copy()
        calls = np.asarray([1, 7, 20, 72, 141], dtype=np.int64)
        committed = np.asarray([False, True, True, True, False])
        counters = np.zeros((5, 32), dtype=np.int64)
        for row, nf in enumerate((30, 210, 429, 1993, 9990)):
            counters[row, 1] = nf
        fixture = {
            "run_label": scalar(plotter.SELECTED_LABEL),
            "alpha": scalar(1.0),
            "discrete_residual_ratio": scalar(0.1),
            "discrete_inner_max_steps": scalar(5),
            "number_of_forces": scalar(9990),
            "successful_joint_ef_evaluations": scalar(9990),
            "optimization_internal_nf": scalar(9990),
            "total_internal_nf_consumed": scalar(9990),
            "total_internal_nf_budget": scalar(9990),
            "total_internal_nf_budget_no_overshoot": scalar(True),
            "total_internal_nf_remaining": scalar(0),
            "total_internal_nf_endpoint_evaluations_excluded": scalar(True),
            "total_internal_nf_endpoint_getstate_attempts_expected": scalar(0),
            "total_internal_nf_terminal_partial_tail_present": scalar(False),
            "total_internal_nf_terminal_partial_tail_quarantined": scalar(False),
            "total_internal_nf_last_complete_state_nf": scalar(9990),
            "calls_per_image": calls,
            "reparam_committed": committed,
            "counter_vectors": counters,
            "stored_path_calls": calls,
            "stored_paths": stored_paths,
            "initial_path": initial,
        }
        rows = []
        for call, index, nf in ((7, 1, 210), (20, 2, 429), (72, 3, 1993)):
            rows.append(
                {
                    "call_per_image": call,
                    "actual_optimization_N_F": nf,
                    "path_sha256": plotter.verifier_dtype_shape_sha256(
                        stored_paths[index]
                    ),
                }
            )
        verification = {
            "eligible_stored_calls_per_image": [1, 7, 20, 72],
            "post_initial_reparam_committed_stored_paths": rows,
        }
        eligible = plotter.collect_selected_committed_paths(
            fixture, verification, actual_nf=9990
        )
        self.assertEqual(set(eligible), {-1, 210, 429, 1993})
        self.assertNotIn(9990, eligible)

    def test_registered_ode_snapshot_reconstructs_the_official_32_image_path(self) -> None:
        contract = json.loads(EVOLUTION_TEMPLATE.read_text(encoding="utf-8"))
        _, ledger, _ = plotter.load_ode_neb_paid_path_ledger(
            EVOLUTION_TEMPLATE, contract["ode_neb_paid_path_ledger"]
        )
        archive, band, metadata = plotter.load_ode_neb_snapshot(
            EVOLUTION_TEMPLATE, contract["ode_neb_snapshots"][0], ledger
        )
        self.assertTrue(archive.is_file())
        self.assertEqual(band.shape, (32, 3132, 3))
        self.assertEqual(metadata["optimization_N_F"], 240)
        self.assertEqual(
            metadata["official_ode_ledger_sha256"],
            "8bfa2c2109fad930d2f6969768be1b6b561a3d82f8aecbc89b9873312a1b250b",
        )

    def test_all_four_registered_ode_snapshots_match_the_official_ledger(self) -> None:
        contract = json.loads(EVOLUTION_TEMPLATE.read_text(encoding="utf-8"))
        _, ledger, _ = plotter.load_ode_neb_paid_path_ledger(
            EVOLUTION_TEMPLATE, contract["ode_neb_paid_path_ledger"]
        )
        observed = []
        for record in contract["ode_neb_snapshots"]:
            _, _, metadata = plotter.load_ode_neb_snapshot(
                EVOLUTION_TEMPLATE, record, ledger
            )
            observed.append(
                (
                    metadata["optimization_N_F"],
                    metadata["batch_id"],
                    metadata["official_paid_path_state_role"],
                )
            )
        self.assertEqual(
            observed,
            [(240, 8, "trial"), (480, 16, "trial"), (990, 33, "trial"), (9990, 333, "trial")],
        )

    def test_ode_snapshot_relabelled_to_a_different_nf_is_rejected(self) -> None:
        contract = json.loads(EVOLUTION_TEMPLATE.read_text(encoding="utf-8"))
        _, ledger, _ = plotter.load_ode_neb_paid_path_ledger(
            EVOLUTION_TEMPLATE, contract["ode_neb_paid_path_ledger"]
        )
        tampered = copy.deepcopy(contract["ode_neb_snapshots"][0])
        tampered["optimization_N_F"] = 270
        tampered["batch_id"] = 9
        with self.assertRaisesRegex(RuntimeError, "official ledger archive"):
            plotter.load_ode_neb_snapshot(EVOLUTION_TEMPLATE, tampered, ledger)

    def test_ode_snapshot_substituted_archive_is_rejected(self) -> None:
        contract = json.loads(EVOLUTION_TEMPLATE.read_text(encoding="utf-8"))
        _, ledger, _ = plotter.load_ode_neb_paid_path_ledger(
            EVOLUTION_TEMPLATE, contract["ode_neb_paid_path_ledger"]
        )
        tampered = copy.deepcopy(contract["ode_neb_snapshots"][0])
        replacement = contract["ode_neb_snapshots"][1]
        tampered["path_archive"] = copy.deepcopy(replacement["path_archive"])
        tampered["official_path32_sha256"] = replacement["official_path32_sha256"]
        with self.assertRaisesRegex(RuntimeError, "official ledger archive"):
            plotter.load_ode_neb_snapshot(EVOLUTION_TEMPLATE, tampered, ledger)

    def test_ode_snapshot_tampered_path_hash_is_rejected(self) -> None:
        contract = json.loads(EVOLUTION_TEMPLATE.read_text(encoding="utf-8"))
        _, ledger, _ = plotter.load_ode_neb_paid_path_ledger(
            EVOLUTION_TEMPLATE, contract["ode_neb_paid_path_ledger"]
        )
        tampered = copy.deepcopy(contract["ode_neb_snapshots"][0])
        tampered["official_path32_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "official path hash mismatch"):
            plotter.load_ode_neb_snapshot(EVOLUTION_TEMPLATE, tampered, ledger)

    def test_formal_legacy_output_is_forbidden(self) -> None:
        identity = {
            "required_output_stem_prefix": "barnase_barstar_ir_ivm_evolution"
        }
        formal = next(iter(plotter.FORBIDDEN_FORMAL_OUTPUTS))
        with self.assertRaisesRegex(RuntimeError, "formal evolution PDF"):
            plotter.validate_output_path(
                formal,
                identity,
                {
                    "pdf_filename": formal.name,
                    "provenance_filename": formal.with_suffix(".json").name,
                },
            )

    def test_new_output_name_and_same_stem_json_are_reserved(self) -> None:
        identity = {
            "required_output_stem_prefix": "barnase_barstar_ir_ivm_evolution"
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "barnase_barstar_ir_ivm_evolution_candidate.pdf"
            output_contract = {
                "pdf_filename": output.name,
                "provenance_filename": output.with_suffix(".json").name,
            }
            resolved, provenance = plotter.validate_output_path(
                output, identity, output_contract
            )
        self.assertEqual(resolved.name, "barnase_barstar_ir_ivm_evolution_candidate.pdf")
        self.assertEqual(provenance.name, "barnase_barstar_ir_ivm_evolution_candidate.json")

    def test_cli_output_filename_must_match_the_final_contract(self) -> None:
        identity = {
            "required_output_stem_prefix": "barnase_barstar_ir_ivm_evolution"
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "barnase_barstar_ir_ivm_evolution_actual.pdf"
            with self.assertRaisesRegex(RuntimeError, "differs from the final figure contract"):
                plotter.validate_output_path(
                    output,
                    identity,
                    {
                        "pdf_filename": "barnase_barstar_ir_ivm_evolution_other.pdf",
                        "provenance_filename": "barnase_barstar_ir_ivm_evolution_other.json",
                    },
                )
            with self.assertRaisesRegex(RuntimeError, "provenance filename differs"):
                plotter.validate_output_path(
                    output,
                    identity,
                    {
                        "pdf_filename": output.name,
                        "provenance_filename": "barnase_barstar_ir_ivm_evolution_wrong.json",
                    },
                )

    def test_duplicate_ivm_or_ode_panel_labels_are_rejected(self) -> None:
        valid = [
            {"ivm_optimization_N_F": 210, "ode_neb_optimization_N_F": 240},
            {"ivm_optimization_N_F": 429, "ode_neb_optimization_N_F": 480},
            {"ivm_optimization_N_F": 1993, "ode_neb_optimization_N_F": 9990},
        ]
        self.assertEqual(
            plotter.validate_panel_contract(valid),
            [(210, 240), (429, 480), (1993, 9990)],
        )
        duplicate_ivm = copy.deepcopy(valid)
        duplicate_ivm[2]["ivm_optimization_N_F"] = 429
        with self.assertRaisesRegex(RuntimeError, "duplicate IVM NoF"):
            plotter.validate_panel_contract(duplicate_ivm)
        duplicate_ode = copy.deepcopy(valid)
        duplicate_ode[2]["ode_neb_optimization_N_F"] = 480
        with self.assertRaisesRegex(RuntimeError, "duplicate ODE-NEB NoF"):
            plotter.validate_panel_contract(duplicate_ode)

    def test_invocation_record_preserves_exact_process_argv(self) -> None:
        raw = ["--contract", "final.json", "--output", "final.pdf"]
        process = ["scripts/plot_bb_irivm_evolution_hashbound.py", *raw]
        record = plotter.invocation_record(raw, process)
        self.assertEqual(record["process_argv_exact"], process)
        self.assertEqual(record["normalized_command"][-4:], raw)
        self.assertTrue(Path(record["python_executable"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
