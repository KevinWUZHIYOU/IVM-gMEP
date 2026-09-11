"""Focused, synthetic tests for the R6 V17 formal public-data contract.

These tests perform no restrained MD and create no campaign ``runs`` tree.
They intentionally express the fail-closed semantics in
``reports/R6_TEST_CONTRACT.md`` even when the current draft implementation has
not implemented a check yet.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pytest


CAMPAIGN = Path(__file__).resolve().parents[1]
SRC = CAMPAIGN / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ala2d_irivm_v17 as core
import formal_contract as contract


def _kind_sequence(
    budget: int,
    overrides: Optional[Mapping[int, str]] = None,
) -> Sequence[str]:
    count = budget // contract.COMPLETE_BAND_NOF + 1
    values = [contract.COMMON_INITIAL_STATE_KIND]
    values.extend([core.STATE_KIND] * (count - 1))
    for index, kind in (overrides or {}).items():
        values[int(index)] = str(kind)
    return values


def synthetic_trajectory(
    *,
    budget: int,
    overrides: Optional[Mapping[int, str]] = None,
) -> Dict[str, object]:
    """Return one internally consistent synthetic public trajectory."""

    kinds = np.asarray(_kind_sequence(budget, overrides))
    count = kinds.size
    nof = np.arange(count, dtype=np.int64) * contract.COMPLETE_BAND_NOF
    paths = np.zeros((count, core.N_PATH, core.DIMENSION), dtype=np.float64)
    base = np.linspace(-1.0, 1.0, core.N_PATH)
    paths[0, :, 0] = base
    paths[0, :, 1] = 0.5 * base
    movement_count = np.zeros(count, dtype=np.int64)
    for index in range(1, count):
        paths[index] = paths[index - 1]
        if str(kinds[index]) == core.STATE_KIND:
            paths[index, 1:-1, 0] += index * 1.0e-7
            movement_count[index] = movement_count[index - 1] + 1
        else:
            movement_count[index] = movement_count[index - 1]

    is_movement = kinds == core.STATE_KIND
    is_hold = np.asarray(
        [False] + [str(kind) != core.STATE_KIND for kind in kinds[1:]],
        dtype=bool,
    )
    is_validation = kinds == core.WARM_VALIDATION_STATE_KIND
    validation_cost = np.where(is_validation, contract.COMPLETE_BAND_NOF, 0).astype(
        np.int64
    )
    fresh_block_count = np.where(is_validation, 10, 0).astype(np.int64)
    metric = np.repeat(
        np.eye(core.DIMENSION, dtype=np.float64)[None, None, :, :],
        count * core.N_INTERNAL,
        axis=0,
    ).reshape(count, core.N_INTERNAL, core.DIMENSION, core.DIMENSION)
    gradient = np.zeros(
        (count, core.N_INTERNAL, core.DIMENSION), dtype=np.float64
    )
    return {
        "implementation_id": np.asarray([core.IMPLEMENTATION_ID] * count),
        "development_only": np.zeros(count, dtype=bool),
        "state_index": np.arange(count, dtype=np.int64),
        "nof_mean_force_incremental": nof,
        "nof_mean_force_absolute_including_shared_initial": nof
        + contract.COMPLETE_BAND_NOF,
        "state_kind": kinds,
        "full_path_unwrapped": paths,
        "M": metric,
        "grad_F": gradient,
        "I_perp_G": np.linspace(3.0, 1.0, count),
        "I_arc": np.linspace(0.3, 0.1, count),
        "Delta_F_barrier": np.linspace(2.0, 2.4, count),
        "segment_length_relative_deviation": np.zeros(count),
        "selected_reparameterization_output_residual": np.zeros(count),
        "hold_resample_selected": is_hold,
        "warm_validation_selected": is_validation,
        "warm_validation_band_mean_force_cost": validation_cost,
        "warm_validation_incremented_movement_count": np.zeros(count, dtype=bool),
        "warm_validation_fresh_raw_block_count": fresh_block_count,
        "warm_validation_used_pooled_source_estimate": np.zeros(
            count, dtype=bool
        ),
        "warm_validation_used_shrinkage_centered_residual": np.zeros(
            count, dtype=bool
        ),
        "paid_action_is_movement": is_movement,
        "movement_count": movement_count,
        "zero_nof_trust_refresh_applied": np.zeros(count, dtype=bool),
    }


def pending_completion(budget: int = 2000) -> Dict[str, object]:
    return {
        "implementation_id": core.IMPLEMENTATION_ID,
        "development_only": False,
        "phase": "VALIDATED_FROZEN_WARM_PENDING_COMMIT",
        "completed": True,
        "budget_prefix_complete": True,
        "requested_prefix_complete": True,
        "budget_fully_spent": True,
        "paused_at_requested_stop": False,
        "pending_next_action": True,
        "pending_validated_commit_requires_another_band": True,
        "pending_warm_status": core.WARM_VALIDATION_PENDING_COMMIT,
        "nof_mean_force_incremental": budget,
        "nof_budget": budget,
        "algorithmic_zero_nof_stop": None,
    }


def _validate(
    trajectory: Mapping[str, object],
    *,
    budget: int,
    completion: Optional[Mapping[str, object]] = None,
):
    return contract.validate_public_trajectory(
        trajectory,
        expected_budget=budget,
        completion=completion,
    )


def test_contract_uses_the_exact_four_v17_public_labels():
    assert contract.COMMON_INITIAL_STATE_KIND == "common_initial"
    assert contract.PAID_STATE_KINDS == (
        "reparameterized_frozen_inner_trial",
        "exact_center_hold_resample",
        "independent_exact_center_warm_validation",
    )
    assert set(contract.PAID_STATE_KINDS) == {
        core.STATE_KIND,
        core.HOLD_STATE_KIND,
        core.WARM_VALIDATION_STATE_KIND,
    }
    assert "reparameterized_inner_trial" not in contract.PAID_STATE_KINDS


def test_all_three_paid_kinds_are_scoreable_on_the_exact_20_nof_grid():
    trajectory = synthetic_trajectory(
        budget=60,
        overrides={
            1: core.STATE_KIND,
            2: core.HOLD_STATE_KIND,
            3: core.WARM_VALIDATION_STATE_KIND,
        },
    )
    terminal = _validate(trajectory, budget=60)
    assert terminal.index == 3
    assert terminal.nof == 60
    assert terminal.state_kind == core.WARM_VALIDATION_STATE_KIND
    np.testing.assert_array_equal(
        trajectory["nof_mean_force_incremental"], [0, 20, 40, 60]
    )
    np.testing.assert_array_equal(
        trajectory["nof_mean_force_absolute_including_shared_initial"],
        [20, 40, 60, 80],
    )
    np.testing.assert_array_equal(trajectory["movement_count"], [0, 1, 1, 1])
    np.testing.assert_array_equal(
        np.asarray(trajectory["full_path_unwrapped"])[1],
        np.asarray(trajectory["full_path_unwrapped"])[2],
    )
    np.testing.assert_array_equal(
        np.asarray(trajectory["full_path_unwrapped"])[2],
        np.asarray(trajectory["full_path_unwrapped"])[3],
    )


@pytest.mark.parametrize(
    "field,index,replacement",
    [
        ("hold_resample_selected", 2, False),
        ("paid_action_is_movement", 2, True),
        ("warm_validation_selected", 3, False),
        ("movement_count", 3, 2),
    ],
)
def test_kind_action_and_movement_fields_cannot_disagree(
    field: str, index: int, replacement: object
):
    trajectory = synthetic_trajectory(
        budget=60,
        overrides={
            1: core.STATE_KIND,
            2: core.HOLD_STATE_KIND,
            3: core.WARM_VALIDATION_STATE_KIND,
        },
    )
    changed = np.asarray(trajectory[field]).copy()
    changed[index] = replacement
    trajectory[field] = changed
    with pytest.raises(contract.FormalContractError):
        _validate(trajectory, budget=60)


@pytest.mark.parametrize(
    "kind,index",
    [
        (core.HOLD_STATE_KIND, 2),
        (core.WARM_VALIDATION_STATE_KIND, 3),
    ],
)
def test_hold_and_validation_must_reuse_the_exact_source_path(
    kind: str, index: int
):
    trajectory = synthetic_trajectory(
        budget=60,
        overrides={2: core.HOLD_STATE_KIND, 3: core.WARM_VALIDATION_STATE_KIND},
    )
    assert str(np.asarray(trajectory["state_kind"])[index]) == kind
    paths = np.asarray(trajectory["full_path_unwrapped"]).copy()
    paths[index, 1, 0] = np.nextafter(paths[index, 1, 0], np.inf)
    trajectory["full_path_unwrapped"] = paths
    with pytest.raises(contract.FormalContractError):
        _validate(trajectory, budget=60)


def test_movement_cannot_masquerade_as_a_byte_identical_hold():
    trajectory = synthetic_trajectory(budget=20)
    paths = np.asarray(trajectory["full_path_unwrapped"]).copy()
    paths[1] = paths[0]
    trajectory["full_path_unwrapped"] = paths
    with pytest.raises(contract.FormalContractError):
        _validate(trajectory, budget=20)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_paid_row",
        "duplicate_nof",
        "wrong_band_cost",
        "legacy_kind",
        "wrong_absolute_nof",
    ],
)
def test_public_ledger_is_exact_contiguous_and_each_paid_row_costs_20(
    mutation: str,
):
    trajectory = synthetic_trajectory(budget=60)
    if mutation == "missing_paid_row":
        for key, value in list(trajectory.items()):
            array = np.asarray(value)
            if array.ndim and array.shape[0] == 4:
                trajectory[key] = np.delete(array, 2, axis=0)
    elif mutation == "duplicate_nof":
        nof = np.asarray(trajectory["nof_mean_force_incremental"]).copy()
        nof[2] = nof[1]
        trajectory["nof_mean_force_incremental"] = nof
    elif mutation == "wrong_band_cost":
        nof = np.asarray(trajectory["nof_mean_force_incremental"]).copy()
        nof[2] = 39
        trajectory["nof_mean_force_incremental"] = nof
    elif mutation == "legacy_kind":
        kinds = np.asarray(trajectory["state_kind"]).copy()
        kinds[2] = "reparameterized_inner_trial"
        trajectory["state_kind"] = kinds
    elif mutation == "wrong_absolute_nof":
        absolute = np.asarray(
            trajectory["nof_mean_force_absolute_including_shared_initial"]
        ).copy()
        absolute[2] += 1
        trajectory["nof_mean_force_absolute_including_shared_initial"] = absolute
    with pytest.raises(contract.FormalContractError):
        _validate(trajectory, budget=60)


@pytest.mark.parametrize(
    "mutation",
    ["missing_identity", "wrong_identity", "development_only"],
)
def test_every_public_row_has_the_exact_formal_v17_identity(mutation: str):
    trajectory = synthetic_trajectory(budget=20)
    if mutation == "missing_identity":
        del trajectory["implementation_id"]
    elif mutation == "wrong_identity":
        identity = np.asarray(trajectory["implementation_id"]).copy()
        identity[-1] = "ala2d-riemannian-v13-unified-immediate-article-reparam-irivm"
        trajectory["implementation_id"] = identity
    else:
        development = np.asarray(trajectory["development_only"]).copy()
        development[-1] = True
        trajectory["development_only"] = development
    with pytest.raises(contract.FormalContractError):
        _validate(trajectory, budget=20)


def test_exact_2000_pending_validation_scores_the_public_source_path():
    trajectory = synthetic_trajectory(
        budget=2000,
        overrides={100: core.WARM_VALIDATION_STATE_KIND},
    )
    paths = np.asarray(trajectory["full_path_unwrapped"])
    assert np.array_equal(paths[100], paths[99])
    future_candidate = paths[100].copy()
    future_candidate[1:-1, 1] += 0.123

    terminal = _validate(
        trajectory,
        budget=2000,
        completion=pending_completion(),
    )
    assert terminal.index == 100
    assert terminal.nof == 2000
    assert terminal.pending_next_action
    assert terminal.state_kind == core.WARM_VALIDATION_STATE_KIND
    np.testing.assert_array_equal(terminal.full_path_unwrapped, paths[99])
    assert not np.array_equal(terminal.full_path_unwrapped, future_candidate)
    assert float(np.asarray(trajectory["I_perp_G"])[100]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("warm_validation_band_mean_force_cost", 0),
        ("warm_validation_fresh_raw_block_count", 0),
        ("warm_validation_used_pooled_source_estimate", True),
        ("warm_validation_used_shrinkage_centered_residual", True),
        ("warm_validation_incremented_movement_count", True),
    ],
)
def test_independent_validation_is_one_fresh_raw_band_only(
    field: str, replacement: object
):
    trajectory = synthetic_trajectory(
        budget=2000,
        overrides={100: core.WARM_VALIDATION_STATE_KIND},
    )
    changed = np.asarray(trajectory[field]).copy()
    changed[100] = replacement
    trajectory[field] = changed
    with pytest.raises(contract.FormalContractError):
        _validate(
            trajectory,
            budget=2000,
            completion=pending_completion(),
        )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("phase", "READY"),
        ("pending_validated_commit_requires_another_band", False),
        ("pending_warm_status", core.WARM_VALIDATION_PENDING_NONE),
        ("requested_prefix_complete", False),
        ("budget_fully_spent", False),
    ],
)
def test_pending_completion_fields_are_one_consistent_state_machine(
    field: str, replacement: object
):
    trajectory = synthetic_trajectory(
        budget=2000,
        overrides={100: core.WARM_VALIDATION_STATE_KIND},
    )
    completion = pending_completion()
    completion[field] = replacement
    with pytest.raises(contract.FormalContractError):
        _validate(trajectory, budget=2000, completion=completion)


def test_pending_validation_path_cannot_be_replaced_by_the_future_candidate():
    trajectory = synthetic_trajectory(
        budget=2000,
        overrides={100: core.WARM_VALIDATION_STATE_KIND},
    )
    paths = np.asarray(trajectory["full_path_unwrapped"]).copy()
    paths[100, 1:-1, 1] += 0.25
    trajectory["full_path_unwrapped"] = paths
    with pytest.raises(contract.FormalContractError):
        _validate(
            trajectory,
            budget=2000,
            completion=pending_completion(),
        )


def test_zero_nof_refresh_is_metadata_on_the_next_paid_row_not_a_public_row():
    trajectory = synthetic_trajectory(budget=40)
    refresh = np.asarray(trajectory["zero_nof_trust_refresh_applied"]).copy()
    refresh[2] = True
    trajectory["zero_nof_trust_refresh_applied"] = refresh
    terminal = _validate(trajectory, budget=40)
    assert terminal.index == 2
    np.testing.assert_array_equal(
        trajectory["nof_mean_force_incremental"], [0, 20, 40]
    )

    duplicated = deepcopy(trajectory)
    for key, value in list(duplicated.items()):
        array = np.asarray(value)
        if array.ndim and array.shape[0] == 3:
            duplicated[key] = np.insert(array, 2, array[1], axis=0)
    # A zero-NoF event cannot appear as a duplicate public checkpoint.
    with pytest.raises(contract.FormalContractError):
        _validate(duplicated, budget=40)
