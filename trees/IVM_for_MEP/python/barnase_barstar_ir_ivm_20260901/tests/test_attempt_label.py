#!/usr/bin/env python3
"""Zero-OpenMM tests for immutable retry identities."""

import unittest

from run_barnase_barstar_ir_ivm_member import member_label


class AttemptLabelTests(unittest.TestCase):
    def test_default_identity_is_unchanged(self):
        self.assertEqual(
            member_label(10, 0.5, 5, 990),
            "bb_irivm_a1e1_r05_m05_inf990_r1",
        )

    def test_retry_identity_is_distinct_and_strict(self):
        self.assertEqual(
            member_label(10, 0.5, 5, 990, "r2"),
            "bb_irivm_a1e1_r05_m05_inf990_r2",
        )
        with self.assertRaisesRegex(ValueError, "attempt tag"):
            member_label(10, 0.5, 5, 990, "retry")


if __name__ == "__main__":
    unittest.main()
