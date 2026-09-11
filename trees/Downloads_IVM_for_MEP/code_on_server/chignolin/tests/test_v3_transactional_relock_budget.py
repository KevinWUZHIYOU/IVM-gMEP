"""Regression checks for the universal V3 relock budget reservation."""

from __future__ import annotations

from run_ivm_early_block_relock_v3_20260729 import (
    immediate_relock_fallback_budget,
)


def test_transactional_hermite_reserves_exactly_one_evaluation():
    assert immediate_relock_fallback_budget(20, True, "hermite_relock") == (
        19,
        1,
    )
    assert immediate_relock_fallback_budget(1, True, "hermite_relock") == (
        0,
        1,
    )
    assert immediate_relock_fallback_budget(0, True, "hermite_relock") == (
        0,
        1,
    )


def test_legacy_and_nonhermite_modes_do_not_change_the_raw_budget():
    assert immediate_relock_fallback_budget(
        20, False, "hermite_relock"
    ) == (20, 0)
    assert immediate_relock_fallback_budget(20, True, "euler") == (20, 0)
    assert immediate_relock_fallback_budget(
        20, True, "relative_defect"
    ) == (20, 0)


if __name__ == "__main__":
    test_transactional_hermite_reserves_exactly_one_evaluation()
    test_legacy_and_nonhermite_modes_do_not_change_the_raw_budget()
    print("2 V3 transactional-relock budget tests passed")
