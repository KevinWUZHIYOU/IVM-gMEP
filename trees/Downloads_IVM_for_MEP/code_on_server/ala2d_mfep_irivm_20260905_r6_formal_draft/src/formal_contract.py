"""Shared, candidate-agnostic contract for the Ala2D V17 formal draft.

The current registration is intentionally not executable.  This module keeps
the runner, acceptance analysis, and plot-data builder on one versioned set of
identities and public-state semantics while leaving the selected controls and
all new random seeds unset until the complete V17 development screen is read.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ala2d_irivm_v17 import (
    HOLD_STATE_KIND,
    IMPLEMENTATION_ID,
    N_INTERNAL,
    STATE_KIND,
    WARM_VALIDATION_PENDING_COMMIT,
    WARM_VALIDATION_STATE_KIND,
)


Array = np.ndarray
PathLike = Union[str, Path]

REGISTRATION_SCHEMA = "ala2d.v17.formal_campaign.registration_draft.v1"
EXECUTABLE_REGISTRATION_SCHEMA = "ala2d.v17.formal_campaign.registration.v1"
DRAFT_STATUS = "DRAFT"
SEALED_STATUS = "SEALED_BEFORE_FIRST_FORMAL_MD"
ACCEPTANCE_SCHEMA = "ala2d.v17.formal_acceptance.v1"
PUBLIC_LEDGER_SCHEMA = "ala2d.v17.formal_public_state_ledger.v1"
ATTEMPT_AUDIT_SCHEMA = "ala2d.v17.formal_md_attempt_audit.v1"

PARAMETER_PLACEHOLDER = "__PENDING_R5_288_RESULT_AND_SELECTION_RECEIPT__"
OPTIMIZATION_SEEDS_PLACEHOLDER = "__PENDING_NEW_OPTIMIZATION_SEEDS__"
DIAGNOSTIC_SEEDS_PLACEHOLDER = "__PENDING_NEW_DIAGNOSTIC_SEEDS__"
BOOTSTRAP_SEED_PLACEHOLDER = "__PENDING_NEW_BOOTSTRAP_SEED__"
CONTOUR_SEED_PLACEHOLDER = "__PENDING_PREREGISTERED_CONTOUR_SEED__"
SHA256_PLACEHOLDER = "__PENDING_FINAL_SHA256_SEAL__"
INPUT_PLACEHOLDER = "__PENDING_SEALED_INPUT_PATH__"
DERIVED_INPUT_PLACEHOLDER = "__PENDING_INPUT_DERIVED_VALUE_AND_HASH_SEAL__"

COMMON_INITIAL_STATE_KIND = "common_initial"
PAID_STATE_KINDS = (
    STATE_KIND,
    HOLD_STATE_KIND,
    WARM_VALIDATION_STATE_KIND,
)
SCOREABLE_STATE_KINDS = (COMMON_INITIAL_STATE_KIND,) + PAID_STATE_KINDS
MEAN_FORCE_VECTOR_DIMENSION = 2
COMPLETE_BAND_NOF = N_INTERNAL
FORMAL_INCREMENTAL_NOF_BUDGET = 2000
FORMAL_CHECKPOINTS = tuple(range(0, FORMAL_INCREMENTAL_NOF_BUDGET + 1, N_INTERNAL))
HOST_GLOBAL_LOCK_RELATIVE_TO_WORKSPACE = ".ala_global_heavy_task.lock"


class FormalContractError(ValueError):
    """Raised when draft, evidence, or trajectory bytes violate the contract."""


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise FormalContractError(f"{key} must be a JSON object")
    return value


def _exact(parent: Mapping[str, object], key: str, expected: object) -> None:
    observed = parent.get(key)
    if observed != expected or type(observed) is not type(expected):
        raise FormalContractError(f"{key} must equal {expected!r}")


def load_json_object(path: PathLike) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalContractError(f"cannot read JSON object from {path}") from error
    if not isinstance(value, dict):
        raise FormalContractError("JSON root must be an object")
    return value


def validate_draft_registration(registration: Mapping[str, object]) -> None:
    """Validate the exact fail-closed shape of the current r6 draft."""

    _exact(registration, "schema_version", REGISTRATION_SCHEMA)
    _exact(registration, "status", DRAFT_STATUS)
    _exact(registration, "formal_evidence", False)
    _exact(registration, "new_restrained_md_authorized", False)
    _exact(registration, "executable", False)

    candidate = _mapping(registration, "candidate")
    _exact(candidate, "selected_parameter", PARAMETER_PLACEHOLDER)
    _exact(candidate, "development_exact_trajectory_count_required", 288)
    _exact(candidate, "selection_receipt_sha256", SHA256_PLACEHOLDER)
    _exact(candidate, "implementation_id", IMPLEMENTATION_ID)

    seeds = _mapping(registration, "seeds")
    _exact(seeds, "optimization", OPTIMIZATION_SEEDS_PLACEHOLDER)
    _exact(seeds, "terminal_diagnostic", DIAGNOSTIC_SEEDS_PLACEHOLDER)
    _exact(seeds, "bootstrap", BOOTSTRAP_SEED_PLACEHOLDER)
    _exact(seeds, "contour", CONTOUR_SEED_PLACEHOLDER)

    execution = _mapping(registration, "execution")
    _exact(execution, "jobs", 1)
    _exact(execution, "lock_kind", "host-global fcntl")
    _exact(
        execution,
        "lock_path_relative_to_workspace",
        HOST_GLOBAL_LOCK_RELATIVE_TO_WORKSPACE,
    )
    _exact(execution, "run_root_must_be_absent", True)
    _exact(execution, "cross_host_execution_authorized", False)
    _exact(execution, "maximum_attempts_per_logical_evaluation", 3)
    _exact(execution, "mutagen_preflight_required", True)
    mutagen = _mapping(execution, "mutagen")
    _exact(mutagen, "endpoints_connected_and_scanned_required", True)
    _exact(mutagen, "paused_must_be_false", True)

    nof = _mapping(registration, "nof")
    _exact(nof, "mean_force_vector_dimension", MEAN_FORCE_VECTOR_DIMENSION)
    _exact(nof, "complete_band", COMPLETE_BAND_NOF)
    _exact(nof, "incremental_budget_per_optimization_run", FORMAL_INCREMENTAL_NOF_BUDGET)
    _exact(nof, "common_initial_incremental_cost", 0)
    _exact(nof, "repeated_center_is_charged_again", True)
    _exact(nof, "zero_nof_refresh_creates_public_state", False)

    public = _mapping(registration, "public_state_contract")
    _exact(public, "common_initial_kind", COMMON_INITIAL_STATE_KIND)
    _exact(public, "paid_state_kinds", list(PAID_STATE_KINDS))
    _exact(public, "acceptance_schema", ACCEPTANCE_SCHEMA)
    _exact(public, "terminal_pending_path_policy", "score_exact_public_source_path_at_2000_without_2020_commit_or_LOCF")

    barrier = _mapping(registration, "barrier_reference")
    _exact(barrier, "value_kcal_per_mol", DERIVED_INPUT_PLACEHOLDER)
    _exact(barrier, "saddle_coordinate_rad", DERIVED_INPUT_PLACEHOLDER)

    plotting = _mapping(registration, "plotting")
    _exact(plotting, "acceptance_schema_required", ACCEPTANCE_SCHEMA)
    _exact(plotting, "method_label", "IVM")
    _exact(plotting, "x_axis", "Number of mean-force evaluations")


def assert_registration_is_not_executable(registration: Mapping[str, object]) -> None:
    """Raise for the current draft and for every partly filled near-draft."""

    if (
        registration.get("schema_version") != EXECUTABLE_REGISTRATION_SCHEMA
        or registration.get("status") != SEALED_STATUS
        or registration.get("formal_evidence") is not True
        or registration.get("new_restrained_md_authorized") is not True
        or registration.get("executable") is not True
    ):
        raise FormalContractError(
            "formal MD is disabled: registration is a DRAFT and lacks a sealed authorization"
        )
    serialized = json.dumps(registration, sort_keys=True)
    unresolved = sorted(set(re.findall(r"__PENDING_[A-Z0-9_]+__", serialized)))
    if unresolved:
        raise FormalContractError(
            "formal MD is disabled: unresolved registration placeholders "
            + ", ".join(unresolved)
        )
    for placeholder in (
        PARAMETER_PLACEHOLDER,
        OPTIMIZATION_SEEDS_PLACEHOLDER,
        DIAGNOSTIC_SEEDS_PLACEHOLDER,
        BOOTSTRAP_SEED_PLACEHOLDER,
        CONTOUR_SEED_PLACEHOLDER,
        SHA256_PLACEHOLDER,
        INPUT_PLACEHOLDER,
        DERIVED_INPUT_PLACEHOLDER,
    ):
        if placeholder in serialized:
            raise FormalContractError(
                f"formal MD is disabled: unresolved registration placeholder {placeholder}"
            )


@dataclass(frozen=True)
class TerminalPublicState:
    index: int
    nof: int
    state_kind: str
    full_path_unwrapped: Array
    pending_next_action: bool


def validate_public_trajectory(
    trajectory: Mapping[str, object],
    *,
    expected_budget: int = FORMAL_INCREMENTAL_NOF_BUDGET,
    completion: Optional[Mapping[str, object]] = None,
) -> TerminalPublicState:
    """Validate an exact V17 prefix and return its actual terminal public path.

    No last-observation-carried-forward substitution is performed.  A paid
    HOLD or independent validation at 2000 is itself the scoreable public
    state.  If validation left a frozen movement pending, the unpurchased 2020
    candidate is deliberately absent from the returned evidence.
    """

    if type(expected_budget) is not int or expected_budget < 0 or expected_budget % N_INTERNAL:
        raise FormalContractError("expected budget must be a nonnegative multiple of 20")
    nof = np.asarray(trajectory.get("nof_mean_force_incremental"), dtype=np.int64)
    indices = np.asarray(trajectory.get("state_index"), dtype=np.int64)
    kinds = np.asarray(trajectory.get("state_kind"))
    paths = np.asarray(trajectory.get("full_path_unwrapped"), dtype=np.float64)
    expected_nof = np.arange(0, expected_budget + N_INTERNAL, N_INTERNAL, dtype=np.int64)
    if not np.array_equal(nof, expected_nof):
        raise FormalContractError("public NoF prefix must be exact 0:20:budget")
    if not np.array_equal(indices, np.arange(expected_nof.size, dtype=np.int64)):
        raise FormalContractError("public state_index must be exact and contiguous")
    if kinds.shape != nof.shape:
        raise FormalContractError("state_kind must align one-for-one with public NoF")
    if paths.shape != (nof.size, N_INTERNAL + 2, MEAN_FORCE_VECTOR_DIMENSION):
        raise FormalContractError("public path tensor has the wrong shape")
    if np.any(~np.isfinite(paths)):
        raise FormalContractError("public paths contain nonfinite coordinates")
    identities = np.asarray(trajectory.get("implementation_id"))
    if identities.shape != nof.shape or np.any(identities.astype(str) != IMPLEMENTATION_ID):
        raise FormalContractError("every public row must carry the exact formal V17 identity")
    development = np.asarray(trajectory.get("development_only"))
    if development.shape != nof.shape or np.any(development != False):  # noqa: E712
        raise FormalContractError("formal public evidence contains a development-only row")
    absolute_nof = np.asarray(
        trajectory.get("nof_mean_force_absolute_including_shared_initial"),
        dtype=np.int64,
    )
    if absolute_nof.shape != nof.shape or not np.array_equal(
        absolute_nof, nof + COMPLETE_BAND_NOF
    ):
        raise FormalContractError("absolute NoF must equal incremental NoF plus the shared band")
    if str(kinds[0]) != COMMON_INITIAL_STATE_KIND:
        raise FormalContractError("state zero is not the registered common initial state")
    bad = [str(value) for value in kinds[1:] if str(value) not in PAID_STATE_KINDS]
    if bad:
        raise FormalContractError(f"unknown paid V17 state kinds: {sorted(set(bad))}")

    expected_holds = np.asarray(
        [False] + [str(value) != STATE_KIND for value in kinds[1:]], dtype=bool
    )
    expected_validations = np.asarray(
        [False]
        + [str(value) == WARM_VALIDATION_STATE_KIND for value in kinds[1:]],
        dtype=bool,
    )
    expected_movements = np.asarray(
        [False] + [str(value) == STATE_KIND for value in kinds[1:]], dtype=bool
    )
    holds = np.asarray(trajectory.get("hold_resample_selected"), dtype=bool)
    validations = np.asarray(
        trajectory.get("warm_validation_selected"), dtype=bool
    )
    movements = np.asarray(
        trajectory.get("paid_action_is_movement"), dtype=bool
    )
    if holds.shape != nof.shape or not np.array_equal(holds, expected_holds):
        raise FormalContractError("HOLD/validation flags disagree with state_kind")
    if validations.shape != nof.shape or not np.array_equal(
        validations, expected_validations
    ):
        raise FormalContractError("independent-validation flags disagree with state_kind")
    if movements.shape != nof.shape or not np.array_equal(
        movements, expected_movements
    ):
        raise FormalContractError("paid movement flags disagree with state_kind")
    validation_cost = np.asarray(
        trajectory.get("warm_validation_band_mean_force_cost"), dtype=np.int64
    )
    fresh_block_count = np.asarray(
        trajectory.get("warm_validation_fresh_raw_block_count"), dtype=np.int64
    )
    used_pooled = np.asarray(
        trajectory.get("warm_validation_used_pooled_source_estimate"), dtype=bool
    )
    used_shrinkage = np.asarray(
        trajectory.get("warm_validation_used_shrinkage_centered_residual"),
        dtype=bool,
    )
    incremented_movement = np.asarray(
        trajectory.get("warm_validation_incremented_movement_count"), dtype=bool
    )
    for name, values in (
        ("warm validation cost", validation_cost),
        ("warm validation fresh blocks", fresh_block_count),
        ("warm validation pooled-source flag", used_pooled),
        ("warm validation shrinkage flag", used_shrinkage),
        ("warm validation movement flag", incremented_movement),
    ):
        if values.shape != nof.shape:
            raise FormalContractError(f"{name} is not aligned to the public ledger")
    if np.any(validation_cost[expected_validations] != COMPLETE_BAND_NOF):
        raise FormalContractError("independent validation must cost exactly one fresh band")
    if np.any(validation_cost[~expected_validations] != 0):
        raise FormalContractError("nonvalidation states claim a validation-band cost")
    if np.any(fresh_block_count[expected_validations] <= 0):
        raise FormalContractError("independent validation lacks fresh raw blocks")
    if np.any(fresh_block_count[~expected_validations] != 0):
        raise FormalContractError("nonvalidation states claim fresh validation blocks")
    if np.any(used_pooled[expected_validations]):
        raise FormalContractError("independent validation used a pooled source estimate")
    if np.any(used_shrinkage[expected_validations]):
        raise FormalContractError("independent validation used P_eff/shrinkage evidence")
    if np.any(incremented_movement[expected_validations]):
        raise FormalContractError("independent validation incremented movement count")
    movement_count = np.asarray(trajectory.get("movement_count"), dtype=np.int64)
    expected_movement_count = np.cumsum(expected_movements.astype(np.int64))
    if movement_count.shape != nof.shape or not np.array_equal(
        movement_count, expected_movement_count
    ):
        raise FormalContractError("movement_count is not the exact cumulative movement ledger")
    for index in range(1, nof.size):
        identical = np.array_equal(paths[index], paths[index - 1])
        if expected_holds[index] and not identical:
            raise FormalContractError("HOLD/validation must reuse the exact source path bytes")
        if expected_movements[index] and identical:
            raise FormalContractError("a paid movement must change the source path bytes")
    refresh = np.asarray(
        trajectory.get("zero_nof_trust_refresh_applied"), dtype=bool
    )
    if refresh.shape != nof.shape or bool(refresh[0]):
        raise FormalContractError("zero-NoF refresh annotations are not aligned to paid states")

    pending = False
    if completion is not None:
        if completion.get("implementation_id") != IMPLEMENTATION_ID:
            raise FormalContractError("completion uses the wrong V17 implementation")
        if completion.get("development_only") is not False:
            raise FormalContractError("formal completion is marked development-only")
        if completion.get("completed") is not True or completion.get("budget_prefix_complete") is not True:
            raise FormalContractError("formal optimization lacks a complete budget prefix")
        if int(completion.get("nof_mean_force_incremental", -1)) != expected_budget:
            raise FormalContractError("completion NoF differs from the registered horizon")
        if int(completion.get("nof_budget", -1)) != expected_budget:
            raise FormalContractError("completion budget differs from the registered horizon")
        for key in (
            "requested_prefix_complete",
            "budget_fully_spent",
        ):
            if completion.get(key) is not True:
                raise FormalContractError(f"completion.{key} must be true")
        if completion.get("paused_at_requested_stop") is not False:
            raise FormalContractError("a complete formal horizon cannot be marked paused")
        pending = bool(completion.get("pending_next_action", False))
        pending_commit = completion.get(
            "pending_validated_commit_requires_another_band"
        )
        pending_status = completion.get("pending_warm_status")
        if pending:
            if str(kinds[-1]) != WARM_VALIDATION_STATE_KIND:
                raise FormalContractError("a pending frozen commit must follow a validation public state")
            if completion.get("phase") != "VALIDATED_FROZEN_WARM_PENDING_COMMIT":
                raise FormalContractError("pending validation completion has the wrong phase")
            if pending_commit is not True:
                raise FormalContractError("pending validation must declare one future paid commit")
            if pending_status != WARM_VALIDATION_PENDING_COMMIT:
                raise FormalContractError("pending validation has the wrong durable status")
        elif pending_commit is True:
            raise FormalContractError("nonpending completion claims a future paid commit")
    return TerminalPublicState(
        index=int(indices[-1]),
        nof=int(nof[-1]),
        state_kind=str(kinds[-1]),
        full_path_unwrapped=np.array(paths[-1], copy=True),
        pending_next_action=pending,
    )


def accepted_report_schema(report: Mapping[str, object]) -> None:
    """Single schema gate shared by acceptance production and plotting."""

    _exact(report, "schema_version", ACCEPTANCE_SCHEMA)
    _exact(report, "implementation_id", IMPLEMENTATION_ID)


__all__ = [
    "ACCEPTANCE_SCHEMA",
    "ATTEMPT_AUDIT_SCHEMA",
    "COMMON_INITIAL_STATE_KIND",
    "COMPLETE_BAND_NOF",
    "DRAFT_STATUS",
    "DERIVED_INPUT_PLACEHOLDER",
    "EXECUTABLE_REGISTRATION_SCHEMA",
    "FORMAL_INCREMENTAL_NOF_BUDGET",
    "FormalContractError",
    "HOST_GLOBAL_LOCK_RELATIVE_TO_WORKSPACE",
    "PAID_STATE_KINDS",
    "PUBLIC_LEDGER_SCHEMA",
    "REGISTRATION_SCHEMA",
    "SCOREABLE_STATE_KINDS",
    "SEALED_STATUS",
    "TerminalPublicState",
    "accepted_report_schema",
    "assert_registration_is_not_executable",
    "load_json_object",
    "validate_draft_registration",
    "validate_public_trajectory",
]
