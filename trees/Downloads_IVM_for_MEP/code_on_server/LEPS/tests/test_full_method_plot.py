import unittest

import numpy as np

from analyze_full_method_comparison import number_of_forces


class TestAllMethodForceAxis(unittest.TestCase):
    def test_internal_force_sum_matches_redundant_neb_ledgers(self):
        counters = np.zeros((3, 22), dtype=np.int64)
        counters[:, 1:-1] = np.asarray((1, 2, 4))[:, None]
        data = {
            "calls_per_image": np.asarray([1, 2, 4]),
            "counter_vector": counters,
            "total_internal_calls": counters[:, 1:-1].sum(axis=1),
            # Repeated API requests are not part of the uncached force cost.
            "force_request_vector": counters + 100,
        }
        np.testing.assert_array_equal(number_of_forces(data), [20, 40, 80])

    def test_string_ledger_excludes_fixed_endpoints(self):
        counters = np.zeros((2, 22), dtype=np.int64)
        counters[0] = 1
        counters[1, 0] = 9
        counters[1, -1] = 11
        counters[1, 1:-1] = np.arange(1, 21)
        data = {
            "calls_per_image": np.asarray([1, 20]),
            "joint_evaluations_history": counters,
        }
        np.testing.assert_array_equal(number_of_forces(data), [20, 210])

    def test_disagreeing_redundant_ledgers_fail_closed(self):
        counters = np.zeros((2, 22), dtype=np.int64)
        counters[:, 1:-1] = 1
        data = {
            "calls_per_image": np.asarray([1, 1]),
            "counter_vector": counters,
            "total_internal_calls": np.asarray([20, 21]),
        }
        with self.assertRaisesRegex(ValueError, "Force ledgers disagree"):
            number_of_forces(data)


if __name__ == "__main__":
    unittest.main()
