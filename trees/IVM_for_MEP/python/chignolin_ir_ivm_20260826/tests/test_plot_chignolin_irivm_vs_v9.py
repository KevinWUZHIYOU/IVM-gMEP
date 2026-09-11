#!/usr/bin/env python3
"""OpenMM- and Matplotlib-free tests for the candidate plot data gates."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from plot_chignolin_irivm_vs_v9 import (  # noqa: E402
    IR_COMPLETE_STATE_DEFINITION,
    complete_ir_mask,
    validate_record_indexed_dense_arrays,
)


class CandidatePlotDataGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.counters = np.asarray(
            [np.full(22, value, dtype=np.int64) for value in range(1, 5)]
        )
        self.total_nf = self.counters.sum(axis=1)
        self.i_perp = np.asarray([8.0, 4.0, 2.0, 1.0])
        self.committed = np.asarray([False, False, True, True])

    def test_complete_mask_is_initial_or_reparam_committed_only(self) -> None:
        mask = complete_ir_mask(
            self.committed, IR_COMPLETE_STATE_DEFINITION
        )
        np.testing.assert_array_equal(mask, [True, False, True, True])

    def test_record_indexed_dense_mapping_rejects_uncommitted_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            validate_record_indexed_dense_arrays(
                record_index=np.asarray([0, 1, 2]),
                paper_total_nf=self.total_nf[[0, 1, 2]],
                i_perp=self.i_perp[[0, 1, 2]],
                selected_counter_vectors=self.counters[[0, 1, 2]],
                complete_state_kind=np.asarray(
                    ["initial", "reparam_committed", "reparam_committed"]
                ),
                reparam_committed=self.committed[[0, 1, 2]],
                raw_counter_vectors=self.counters,
                raw_total_nf=self.total_nf,
                raw_i_perp=self.i_perp,
                raw_reparam_committed=self.committed,
            )

    def test_record_indexed_dense_mapping_accepts_exact_committed_records(self) -> None:
        validate_record_indexed_dense_arrays(
            record_index=np.asarray([0, 2, 3]),
            paper_total_nf=self.total_nf[[0, 2, 3]],
            i_perp=self.i_perp[[0, 2, 3]],
            selected_counter_vectors=self.counters[[0, 2, 3]],
            complete_state_kind=np.asarray(
                ["initial", "reparam_committed", "reparam_committed"]
            ),
            reparam_committed=self.committed[[0, 2, 3]],
            raw_counter_vectors=self.counters,
            raw_total_nf=self.total_nf,
            raw_i_perp=self.i_perp,
            raw_reparam_committed=self.committed,
        )


if __name__ == "__main__":
    unittest.main()
