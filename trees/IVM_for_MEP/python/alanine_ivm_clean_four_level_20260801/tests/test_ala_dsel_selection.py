from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from ala_core import clip_max_atomic_displacement, project_normal
from run_ivm_v9_spring_projected_production_20260730 import (
    ProjectedLBFGSMemory,
    _ala_dsel_legacy_uses_curvature,
    _score_ala_curve_proposal_after_geometry,
    _screen_ala_dsel_band,
    _select_ala_dsel_direction,
)
import run_ivm_v9_spring_projected_production_20260730 as runner


class AlaDselSelectionTests(unittest.TestCase):
    def test_screen_preserves_legacy_cap_bytes(self):
        tangent = np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        raw = np.asarray([[0.0, 3.0, 0.0, 0.0, 4.0, 0.0]])
        gradient = -raw.copy()
        expected = project_normal(
            clip_max_atomic_displacement(raw[0], 0.2)[None, :], tangent
        )
        proposal = _screen_ala_dsel_band(
            "LBFGS", raw, gradient, tangent, 0.2
        )
        self.assertTrue(proposal["valid"])
        np.testing.assert_array_equal(proposal["step"], expected)

    def test_exact_tie_selects_h0_object(self):
        tangent = np.asarray([[1.0, 0.0, 0.0]])
        raw = np.asarray([[0.0, 2.0, 0.0]])
        gradient = -raw.copy()
        h0 = _screen_ala_dsel_band("H0", raw, gradient, tangent, 0.1)
        lbfgs = _screen_ala_dsel_band(
            "LBFGS", raw.copy(), gradient, tangent, 0.1
        )
        self.assertEqual(h0["capped_decrease"], lbfgs["capped_decrease"])
        self.assertIs(_select_ala_dsel_direction(h0, lbfgs), h0)

    def test_capped_not_raw_decrease_selects_winner(self):
        gradient = np.asarray([[0.0, -1.0, 0.0, 0.0, -1.0, 0.0]])
        tangent = np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        h0 = _screen_ala_dsel_band(
            "H0",
            [[0.0, 100.0, 0.0, 0.0, 0.0, 0.0]],
            gradient,
            tangent,
            0.1,
        )
        lbfgs = _screen_ala_dsel_band(
            "LBFGS",
            [[0.0, 2.0, 0.0, 0.0, 2.0, 0.0]],
            gradient,
            tangent,
            0.1,
        )
        self.assertGreater(h0["raw_decrease"], lbfgs["raw_decrease"])
        self.assertGreater(
            lbfgs["capped_decrease"], h0["capped_decrease"]
        )
        self.assertEqual(
            _select_ala_dsel_direction(h0, lbfgs)["label"], "LBFGS"
        )

    def test_one_invalid_falls_back_and_two_invalid_fail(self):
        tangent = np.asarray([[1.0, 0.0, 0.0]])
        gradient = np.asarray([[0.0, 1.0, 0.0]])
        invalid = _screen_ala_dsel_band(
            "H0", [[0.0, 1.0, 0.0]], gradient, tangent, 1.0
        )
        valid = _screen_ala_dsel_band(
            "LBFGS", [[0.0, -1.0, 0.0]], gradient, tangent, 1.0
        )
        self.assertIs(_select_ala_dsel_direction(invalid, valid), valid)
        with self.assertRaisesRegex(RuntimeError, "Both Ala DSEL"):
            _select_ala_dsel_direction(invalid, dict(invalid))

    def test_empty_outer_memory_is_an_exact_no_curvature_tie(self):
        tangent = np.asarray([1.0, 0.0, 0.0])
        memory = ProjectedLBFGSMemory(1, 7)
        self.assertFalse(
            _ala_dsel_legacy_uses_curvature([], memory, 0, tangent)
        )
        memory.pairs[0].append(
            (
                np.asarray([0.0, 1.0, 0.0]),
                np.asarray([0.0, 2.0, 0.0]),
            )
        )
        self.assertTrue(
            _ala_dsel_legacy_uses_curvature([], memory, 0, tangent)
        )

    def test_curve_selection_scores_post_geometry_actual_step(self):
        reference = np.zeros((3, 3), dtype=float)
        current = reference.copy()
        displacement = np.zeros((1, 3), dtype=float)
        residual = np.asarray([[0.0, -1.0, 0.0]])
        tangents = np.asarray([[1.0, 0.0, 0.0]])
        h0 = {
            "label": "H0",
            "valid": True,
            "error": "none",
            "step": np.asarray([[0.0, 2.0, 0.0]]),
            "raw_decrease": 2.0,
            "capped_decrease": 2.0,
        }
        lbfgs = {
            "label": "LBFGS",
            "valid": True,
            "error": "none",
            "step": np.asarray([[0.0, 1.0, 0.0]]),
            "raw_decrease": 1.0,
            "capped_decrease": 1.0,
        }

        def fake_spring(path):
            projected = np.asarray(path, dtype=float).copy()
            projected[1, 1] = (
                0.1 if projected[1, 1] > 1.5 else 0.8
            )
            return projected, {"zero_call_test": True}

        with mock.patch.object(
            runner,
            "central_spring_force_equilibrium",
            side_effect=fake_spring,
        ):
            h0_scored = _score_ala_curve_proposal_after_geometry(
                h0,
                reference,
                current,
                displacement,
                residual,
                tangents,
            )
            lbfgs_scored = _score_ala_curve_proposal_after_geometry(
                lbfgs,
                reference,
                current,
                displacement,
                residual,
                tangents,
            )
        self.assertAlmostEqual(h0_scored["post_geometry_decrease"], 0.1)
        self.assertAlmostEqual(
            lbfgs_scored["post_geometry_decrease"], 0.8
        )
        self.assertEqual(
            _select_ala_dsel_direction(h0_scored, lbfgs_scored)["label"],
            "LBFGS",
        )


if __name__ == "__main__":
    unittest.main()
