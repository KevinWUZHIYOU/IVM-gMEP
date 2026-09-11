#!/usr/bin/env python3
"""Matplotlib-free data-gate tests for the parameter-effect figure."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from plot_chignolin_irivm_parameter_effects import (  # noqa: E402
    EXPECTED_ALPHA,
    EXPECTED_M_MAX,
    EXPECTED_R,
    observation_from_row,
    parse_bool,
    select_control_panels,
)


def make_row(alpha, r, m_max, reached=False, nf=1200):
    threshold = 0.21242215133173672
    last_i_perp = 0.18 if reached else 0.30
    row = {
        "parameter_setting_id": (
            f"alpha={alpha}|r={r}|m_max={m_max}|internal_nf_budget=1500"
        ),
        "alpha": str(alpha),
        "residual_ratio_r": str(r),
        "inner_max_steps_m_max": str(m_max),
        "internal_nf_budget": "1500",
        "eta": "1.1",
        "i_perp_threshold": str(threshold),
        "analysis_status": "reached" if reached else "not_reached",
        "audit_failure_reason": "",
        "first_threshold_reached": str(reached),
        "first_threshold_state_index": "5" if reached else "",
        "first_threshold_state_kind": "reparam_committed" if reached else "",
        "first_threshold_total_nf": str(nf - 100) if reached else "",
        "first_threshold_i_perp": "0.18" if reached else "",
        "last_complete_reparam_state_available": "True",
        "last_complete_reparam_state_index": "7",
        "last_complete_reparam_total_nf": str(nf),
        "last_complete_reparam_i_perp": str(last_i_perp),
        "last_usable_state_kind": "reparam_committed",
        "last_usable_state_index": "7",
        "last_usable_total_nf": str(nf),
        "last_usable_i_perp": str(last_i_perp),
        "right_censoring_total_nf": "" if reached else str(nf),
        "terminal_total_nf": "1502",
        "terminal_partial_tail_internal_nf": str(1502 - nf),
        "terminal_partial_tail_quarantined": "True",
    }
    return row


class ParameterEffectDataGateTest(unittest.TestCase):
    def test_parse_bool_is_strict(self):
        self.assertTrue(parse_bool("True"))
        self.assertFalse(parse_bool("false"))
        with self.assertRaisesRegex(ValueError, "must be True or False"):
            parse_bool("yes")

    def test_selects_only_registered_one_factor_controls(self):
        rows = []
        seen = set()
        for alpha in EXPECTED_ALPHA:
            row = make_row(alpha, 0.5, 5)
            rows.append(row)
            seen.add((float(alpha), 0.5, 5))
        for r in EXPECTED_R:
            key = (10.0, float(r), 5)
            if key not in seen:
                rows.append(make_row(*key))
                seen.add(key)
        for m_max in EXPECTED_M_MAX:
            key = (10.0, 0.5, int(m_max))
            if key not in seen:
                rows.append(make_row(*key))
                seen.add(key)
        # This interaction-screen row must not enter any 1500-NF control panel.
        extra = make_row(10, 0.3, 8)
        extra["internal_nf_budget"] = "1000"
        rows.append(extra)

        panels = select_control_panels(rows)
        self.assertEqual([panel.key for panel in panels], ["alpha", "r", "m_max"])
        self.assertEqual(panels[0].values, tuple(EXPECTED_ALPHA))
        self.assertEqual(panels[1].values, tuple(EXPECTED_R))
        self.assertEqual(panels[2].values, tuple(float(x) for x in EXPECTED_M_MAX))

    def test_reached_row_uses_first_committed_passage(self):
        observation = observation_from_row(
            make_row(10, 0.5, 5, reached=True, nf=1200)
        )
        self.assertTrue(observation.reached)
        self.assertEqual(observation.number_of_forces, 1100)
        self.assertEqual(observation.state_kind, "reparam_committed")

    def test_non_reaching_row_is_a_lower_bound_at_last_complete_reparam(self):
        observation = observation_from_row(make_row(1, 0.5, 5, nf=1466))
        self.assertFalse(observation.reached)
        self.assertEqual(observation.number_of_forces, 1466)
        self.assertEqual(observation.state_kind, "reparam_committed")

    def test_non_reaching_row_cannot_carry_a_fake_first_passage(self):
        row = make_row(1, 0.5, 5, nf=1466)
        row["first_threshold_total_nf"] = "900"
        with self.assertRaisesRegex(ValueError, "must be blank"):
            observation_from_row(row)

    def test_censoring_must_equal_last_complete_reparam_nf(self):
        row = make_row(1, 0.5, 5, nf=1466)
        row["right_censoring_total_nf"] = "1502"
        with self.assertRaisesRegex(ValueError, "last complete reparam NF"):
            observation_from_row(row)


if __name__ == "__main__":
    unittest.main()
