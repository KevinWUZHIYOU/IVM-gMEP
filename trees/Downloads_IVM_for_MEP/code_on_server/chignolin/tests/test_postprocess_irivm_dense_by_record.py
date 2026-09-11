#!/usr/bin/env python3
"""OpenMM-free tests for IR-IVM dense-record eligibility and mapping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from postprocess_irivm_dense_by_record import (  # noqa: E402
    COMPLETE_STATE_DEFINITION,
    build_record_selection,
)


class RecordSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        n_records = 5
        self.counter_vectors = np.asarray(
            [
                np.full(22, value, dtype=np.int64)
                for value in range(1, n_records + 1)
            ]
        )
        self.i_perp = np.asarray([10.0, 5.0, 2.0, 1.0, 0.5])
        self.reparam_committed = np.asarray(
            [False, False, True, False, True]
        )
        self.calls = np.arange(1, n_records + 1, dtype=np.int64)
        self.paths = np.arange(
            n_records * 22 * 6, dtype=float
        ).reshape(n_records, 22, 6)

    def test_maps_only_initial_and_committed_records_by_record_index(self) -> None:
        selected = build_record_selection(
            counter_vectors=self.counter_vectors,
            i_perp=self.i_perp,
            reparam_committed=self.reparam_committed,
            record_indices=[0, 2, 4],
            stored_paths=self.paths,
            complete_state_definition=COMPLETE_STATE_DEFINITION,
            calls_per_image=self.calls,
            stored_path_calls=self.calls,
        )
        np.testing.assert_array_equal(selected["record_index"], [0, 2, 4])
        np.testing.assert_array_equal(
            selected["paper_total_nf"], [22, 66, 110]
        )
        np.testing.assert_allclose(selected["i_perp"], [10.0, 2.0, 0.5])
        np.testing.assert_array_equal(
            selected["complete_state_kind"],
            ["initial", "reparam_committed", "reparam_committed"],
        )
        np.testing.assert_array_equal(selected["paths"], self.paths[[0, 2, 4]])

    def test_rejects_uncommitted_intermediate_record(self) -> None:
        with self.assertRaisesRegex(ValueError, "neither the initial state"):
            build_record_selection(
                counter_vectors=self.counter_vectors,
                i_perp=self.i_perp,
                reparam_committed=self.reparam_committed,
                record_indices=[0, 1, 2],
                stored_paths=self.paths,
                complete_state_definition=COMPLETE_STATE_DEFINITION,
                calls_per_image=self.calls,
                stored_path_calls=self.calls,
            )


if __name__ == "__main__":
    unittest.main()
