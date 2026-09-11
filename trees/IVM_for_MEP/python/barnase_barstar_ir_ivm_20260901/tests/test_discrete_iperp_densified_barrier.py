#!/usr/bin/env python3
"""Zero-OpenMM tests for the BB principal-diagnostic evaluator."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_discrete_iperp_densified_barrier import (  # noqa: E402
    ALPHA_SCOPE_R2_SHA256,
    CONTRACT_SCHEMA,
    COORDINATE_SPACE,
    DIMENSION,
    GRID_FORMULA,
    INTERPOLATION,
    METRIC_AMENDMENT_SHA256,
    N_DENSE_POINTS,
    N_IMAGES,
    SAMPLING,
    _validate_contract,
    load_eligible_rows,
    sha256_array,
    sha256_file,
    uniform_arclength_resample,
)
from select_alpha_from_principal_diagnostics import (  # noqa: E402
    select_alpha_records,
)
from select_alpha_from_confirmation_diagnostics import (  # noqa: E402
    select_confirmation_records,
)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class PrincipalDiagnosticTest(unittest.TestCase):
    def test_uniform_arclength_grid_has_exact_count_and_endpoints(self) -> None:
        path = np.zeros((N_IMAGES, DIMENSION), dtype=np.float64)
        path[:, 0] = 5.0 * np.linspace(0.0, 1.0, N_IMAGES) ** 1.4
        dense, geometry = uniform_arclength_resample(path)
        self.assertEqual(dense.shape, (N_DENSE_POINTS, DIMENSION))
        np.testing.assert_array_equal(dense[0], path[0])
        np.testing.assert_array_equal(dense[-1], path[-1])
        spacing = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        np.testing.assert_allclose(spacing, spacing[0], rtol=4e-13, atol=4e-13)
        self.assertAlmostEqual(geometry["total_arclength_A"], 5.0)

    def test_zero_length_image_segment_is_rejected(self) -> None:
        path = np.zeros((N_IMAGES, DIMENSION), dtype=np.float64)
        path[:, 0] = np.arange(N_IMAGES, dtype=np.float64)
        path[9] = path[8]
        with self.assertRaisesRegex(ValueError, "positive finite length"):
            uniform_arclength_resample(path)

    def test_only_fully_reparameterized_states_are_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = np.zeros((N_IMAGES, DIMENSION), dtype=np.float64)
            common[:, 0] = np.arange(N_IMAGES, dtype=np.float64)
            committed = common.copy()
            committed[1:-1, 1] = 0.25
            partial = committed.copy()
            partial[1:-1, 2] = 0.5
            common_file = root / "common.npz"
            np.savez_compressed(common_file, initial_path=common)

            counters = np.zeros((4, N_IMAGES), dtype=np.int64)
            for history_index, count in enumerate((1, 2, 3, 4)):
                counters[history_index, 1:-1] = count
            result = root / "result.npz"
            np.savez_compressed(
                result,
                run_label=np.asarray("synthetic_alpha_1"),
                initial_path=common,
                stored_paths=np.asarray([common, committed, partial]),
                stored_path_calls=np.asarray([1, 3, 4], dtype=np.int64),
                calls_per_image=np.asarray([1, 2, 3, 4], dtype=np.int64),
                counter_vectors=counters,
                i_perp=np.asarray([10.0, 8.0, 5.0, 4.0]),
                reparam_committed=np.asarray([False, False, True, False]),
                number_of_forces=np.asarray(120),
                optimization_internal_nf=np.asarray(120),
                total_internal_nf_budget=np.asarray(120),
                total_internal_nf_consumed=np.asarray(120),
                total_internal_nf_remaining=np.asarray(0),
                total_internal_nf_budget_stop_reason=np.asarray("exact"),
                total_internal_nf_budget_exhausted=np.asarray(True),
                total_internal_nf_budget_no_overshoot=np.asarray(True),
                total_internal_nf_full_band_preflight_zero_step_stops=np.asarray(0),
                total_internal_nf_complete_band_budget_stop_count=np.asarray(0),
                total_internal_nf_terminal_partial_tail_present=np.asarray(True),
                total_internal_nf_terminal_partial_tail_quarantined=np.asarray(True),
            )
            audit = root / "result.audit.json"
            write_json(
                audit,
                {
                    "consistent": True,
                    "backend_raw_joint_attempts": 120,
                    "backend_raw_joint_evaluations": 120,
                    "sum_image_joint_evaluations": 120,
                    "optimization_internal_nf": 120,
                    "number_of_forces": 120,
                    "endpoint_getstate_attempts": 0,
                    "endpoint_joint_ef_cache_misses": 0,
                    "endpoint_inherited_joint_ef_cache_entries": 2,
                },
            )
            contract_path = root / "contract.json"
            contract = {
                "schema": CONTRACT_SCHEMA,
                "diagnostic_protocol": {
                    "coordinate_space": COORDINATE_SPACE,
                    "interpolation": INTERPOLATION,
                    "sampling": SAMPLING,
                    "number_of_points": N_DENSE_POINTS,
                    "include_both_endpoints": True,
                    "grid_formula": GRID_FORMULA,
                    "force_evaluations_per_path": 0,
                    "optimization_N_F_contribution": 0,
                },
                "bindings": {
                    "metric_policy_amendment": {
                        "path": str(
                            ROOT
                            / "registrations"
                            / "USER_METRIC_POLICY_AMENDMENT_20260901_R1.json"
                        ),
                        "sha256": METRIC_AMENDMENT_SHA256,
                    },
                    "alpha_scope_R2": {
                        "path": str(
                            ROOT
                            / "registrations"
                            / "USER_ALPHA_SCOPE_AMENDMENT_20260901_R2.json"
                        ),
                        "sha256": ALPHA_SCOPE_R2_SHA256,
                    },
                },
                "common_initial_path": {
                    "path": str(common_file),
                    "file_sha256": sha256_file(common_file),
                    "array_sha256": sha256_array(common),
                },
                "topology": {"path": "unused.pdb", "sha256": "unused"},
                "runs": [
                    {
                        "label": "synthetic_alpha_1",
                        "alpha": 1.0,
                        "r": 0.5,
                        "m_max": 5,
                        "registered_optimization_N_F": 120,
                        "source_result": str(result),
                        "source_result_sha256": sha256_file(result),
                        "source_audit": str(audit),
                        "source_audit_sha256": sha256_file(audit),
                        "eligible_committed_calls": [3],
                    }
                ],
            }
            write_json(contract_path, contract)

            validated = _validate_contract(contract_path)
            recovered_common, initial_i, initial_nf, rows = load_eligible_rows(
                contract_path, validated
            )
            np.testing.assert_array_equal(recovered_common, common)
            self.assertEqual(initial_i, 10.0)
            self.assertEqual(initial_nf, 30)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["calls_per_image"], 3)
            self.assertEqual(rows[0]["optimization_N_F"], 90)
            self.assertEqual(rows[0]["discrete_I_perp"], 5.0)
            np.testing.assert_array_equal(rows[0]["path"], committed)

    def test_import_is_openmm_free(self) -> None:
        self.assertNotIn("openmm", sys.modules)

    def test_alpha_selection_uses_both_raw_diagnostics_on_one_state(self) -> None:
        labels = np.asarray(
            [
                "common_initial_path",
                "a01",
                "a01",
                "a1",
                "a10",
                "a100",
                "a1000",
            ]
        )
        alpha = np.asarray([np.nan, 0.1, 0.1, 1.0, 10.0, 100.0, 1000.0])
        nf = np.asarray([30, 90, 360, 360, 360, 360, 300])
        i_perp = np.asarray([100.0, 20.0, 8.0, 11.0, 12.0, 14.0, 18.0])
        delta_e = np.asarray([50.0, 6.0, 20.0, 8.0, 7.0, 5.0, 4.0])
        normalization, all_states, ordered = select_alpha_records(
            labels=labels,
            alpha=alpha,
            optimization_nf=nf,
            discrete_i_perp=i_perp,
            densified_delta_e=delta_e,
        )
        self.assertEqual(normalization["discrete_I_perp"], 100.0)
        self.assertEqual(normalization["densified_Delta_E_kcal_per_mol"], 50.0)
        self.assertEqual(len(all_states), 6)
        selected_a01 = next(row for row in ordered if row["alpha"] == 0.1)
        self.assertEqual(selected_a01["optimization_N_F"], 90)
        self.assertEqual(selected_a01["discrete_I_perp"], 20.0)
        self.assertEqual(selected_a01["densified_Delta_E_kcal_per_mol"], 6.0)
        self.assertEqual(len(ordered), 5)

    def test_alpha_confirmation_retains_two_by_same_state_joint_quantity(self) -> None:
        labels = np.asarray(
            ["common_initial_path", "a1", "a1", "a01", "a10"]
        )
        alpha = np.asarray([np.nan, 1.0, 1.0, 0.1, 10.0])
        nf = np.asarray([30, 360, 900, 900, 900])
        i_perp = np.asarray([100.0, 4.0, 1.0, 3.0, 2.0])
        delta_e = np.asarray([50.0, 2.0, 20.0, 4.0, 5.0])
        normalization, all_states, ordered = select_confirmation_records(
            labels=labels,
            alpha=alpha,
            optimization_nf=nf,
            discrete_i_perp=i_perp,
            densified_delta_e=delta_e,
        )
        self.assertEqual(normalization["discrete_I_perp"], 100.0)
        self.assertEqual(normalization["densified_Delta_E_kcal_per_mol"], 50.0)
        self.assertEqual(len(all_states), 4)
        self.assertEqual(ordered[0]["alpha"], 1.0)
        self.assertEqual(ordered[0]["optimization_N_F"], 360)
        self.assertEqual(ordered[1]["alpha"], 0.1)
        self.assertEqual(ordered[2]["alpha"], 10.0)


if __name__ == "__main__":
    unittest.main()
