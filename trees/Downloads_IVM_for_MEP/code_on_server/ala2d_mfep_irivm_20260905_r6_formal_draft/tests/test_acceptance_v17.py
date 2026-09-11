"""V17-only acceptance tests; all evidence is synthetic and spends zero NoF."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ala2d_irivm_v17 import (  # noqa: E402
    HOLD_STATE_KIND,
    IMPLEMENTATION_ID,
    STATE_KIND,
    WARM_VALIDATION_STATE_KIND,
)
from campaign_acceptance import (  # noqa: E402
    AcceptanceInputError,
    evaluate_campaign_acceptance,
    select_exact_paid_public_state,
)
from formal_contract import ACCEPTANCE_SCHEMA, PAID_STATE_KINDS  # noqa: E402


def _registration():
    value = json.loads(
        (ROOT / "registrations/V17_FORMAL_CAMPAIGN_DRAFT.json").read_text(
            encoding="utf-8"
        )
    )
    value = deepcopy(value)
    value["candidate"]["selected_parameter"] = {
        "parameter_id": "synthetic-v17-candidate",
        "alpha": 1.0e-4,
        "r": 0.7,
        "m_max": 4,
    }
    value["seeds"]["optimization"] = [9001, 9002, 9003]
    value["barrier_reference"]["value_kcal_per_mol"] = 1.0
    value["barrier_reference"]["saddle_coordinate_rad"] = [0.5, 0.25]
    guard = value["acceptance"]["correct_path_guard"]
    guard["start_endpoint_rad"] = [0.0, 0.0]
    guard["target_endpoint_rad"] = [1.0, 0.5]
    return value


def _path():
    t = np.linspace(0.0, 1.0, 22)
    return np.column_stack((t, 0.5 * t))


def _trajectory(terminal_kind):
    nof = np.arange(0, 2001, 20, dtype=np.int64)
    count = nof.size
    base_path = _path()
    paths = np.repeat(base_path[None, :, :], count, axis=0)
    kinds = np.asarray(
        ["common_initial"] + [STATE_KIND] * (count - 1), dtype="U64"
    )
    # Put all three paid V17 semantics into the public ledger, including a
    # terminal state that may not be movement.
    kinds[7] = HOLD_STATE_KIND
    kinds[11] = WARM_VALIDATION_STATE_KIND
    kinds[-1] = terminal_kind
    for index in range(1, count):
        paths[index] = paths[index - 1]
        if kinds[index] == STATE_KIND:
            paths[index, 1:-1, 0] += 1.0e-8
    holds = np.asarray(
        [False] + [kind != STATE_KIND for kind in kinds[1:]], dtype=bool
    )
    validations = kinds == WARM_VALIDATION_STATE_KIND
    movements = np.asarray(
        [False] + [kind == STATE_KIND for kind in kinds[1:]], dtype=bool
    )
    metric = np.repeat(np.eye(2)[None, None, :, :], count * 20, axis=0)
    metric = metric.reshape(count, 20, 2, 2)
    deviations = []
    for path in paths:
        lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        deviations.append(float(np.max(np.abs(lengths / np.mean(lengths) - 1.0))))
    return {
        "implementation_id": np.asarray([IMPLEMENTATION_ID] * count),
        "development_only": np.zeros(count, dtype=bool),
        "state_index": np.arange(count, dtype=np.int64),
        "nof_mean_force_incremental": nof,
        "nof_mean_force_absolute_including_shared_initial": nof + 20,
        "state_kind": kinds,
        "hold_resample_selected": holds,
        "warm_validation_selected": validations,
        "warm_validation_band_mean_force_cost": validations.astype(np.int64) * 20,
        "warm_validation_fresh_raw_block_count": validations.astype(np.int64) * 10,
        "warm_validation_used_pooled_source_estimate": np.zeros(count, dtype=bool),
        "warm_validation_used_shrinkage_centered_residual": np.zeros(count, dtype=bool),
        "warm_validation_incremented_movement_count": np.zeros(count, dtype=bool),
        "paid_action_is_movement": movements,
        "movement_count": np.cumsum(movements.astype(np.int64)),
        "zero_nof_trust_refresh_applied": np.zeros(count, dtype=bool),
        "full_path_unwrapped": paths,
        "M": metric,
        "I_perp_G": np.full(count, 0.1),
        "I_arc": np.zeros(count),
        "Delta_F_barrier": np.ones(count),
        "segment_length_relative_deviation": np.asarray(deviations),
        "selected_reparameterization_output_residual": np.concatenate(
            ([np.nan], np.zeros(count - 1))
        ),
    }


def _baseline():
    nof = np.arange(0, 2001, 20, dtype=np.int64)
    count = nof.size
    return {
        "nof_mean_force_incremental": nof,
        "I_perp_G": np.full(count, 0.2),
        "I_arc": np.full(count, 0.1),
        "Delta_F_barrier": np.full(count, 1.2),
    }


def test_acceptance_producer_emits_the_shared_v17_schema_for_all_paid_terminal_kinds():
    registration = _registration()
    terminal_kinds = PAID_STATE_KINDS
    trajectories = {
        seed: _trajectory(kind)
        for seed, kind in zip(registration["seeds"]["optimization"], terminal_kinds)
    }
    report = evaluate_campaign_acceptance(
        trajectories,
        _baseline(),
        _path(),
        registration,
        terminal_diagnostics=None,
    )
    assert report["schema_version"] == ACCEPTANCE_SCHEMA
    assert report["analysis"] == ACCEPTANCE_SCHEMA
    assert report["implementation_id"] == IMPLEMENTATION_ID
    assert report["candidate_parameter_id"] == "synthetic-v17-candidate"
    assert report["optimization_seeds"] == [9001, 9002, 9003]
    assert report["cost_history"]["pass"] is True
    assert report["correct_path"]["pass"] is True
    assert [item["selected_state_kind"] for item in report["correct_path"]["seeds"]] == list(terminal_kinds)
    assert report["decision"] == "pending_terminal_diagnostics"
    assert report["pass"] is None


@pytest.mark.parametrize("kind", PAID_STATE_KINDS)
def test_exact_anchor_selector_accepts_each_paid_v17_state_without_locf(kind):
    trajectory = _trajectory(STATE_KIND)
    trajectory["state_kind"][50] = kind
    trajectory["hold_resample_selected"][50] = kind != STATE_KIND
    assert select_exact_paid_public_state(trajectory, 1000) == 50


def test_exact_anchor_selector_rejects_legacy_single_state_identity():
    trajectory = _trajectory(STATE_KIND)
    trajectory["state_kind"][50] = "reparameterized_inner_trial"
    with pytest.raises(AcceptanceInputError, match="three registered V17 paid states"):
        select_exact_paid_public_state(trajectory, 1000)


def test_acceptance_rejects_an_incomplete_prefix_instead_of_carrying_forward():
    registration = _registration()
    trajectories = {
        seed: _trajectory(STATE_KIND)
        for seed in registration["seeds"]["optimization"]
    }
    for key, value in list(trajectories[9001].items()):
        if isinstance(value, np.ndarray) and value.shape[:1] == (101,):
            trajectories[9001][key] = np.delete(value, 50, axis=0)
    with pytest.raises(AcceptanceInputError, match="exact V17 public prefix"):
        evaluate_campaign_acceptance(
            trajectories,
            _baseline(),
            _path(),
            registration,
            terminal_diagnostics=None,
        )
