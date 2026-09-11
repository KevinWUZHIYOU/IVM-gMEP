#!/usr/bin/env python3
"""Shared exact total-internal-NF budget for Ala runner shims.

The production Ala runner's historical ``--budget`` option is a *per-image*
limit.  This module adds an orthogonal run-global limit over successful
energy/force cache misses of the 20 internal images.  Endpoint evaluations are
deliberately excluded, matching ``total_internal_joint_evaluations`` in the
saved trajectory.

The limiter is installed before the production runner creates its evaluators.
It blocks an uncached internal-image query before OpenMM is called, so the
declared total cannot overshoot.  Cached queries remain available at the
boundary.  A terminal partially evaluated proposal is charged but is never
promoted to a complete outer state; the persisted metadata identifies the last
``initial OR reparam_committed`` state and the quarantined charged tail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


TOTAL_INTERNAL_NF_FLAG = "--total-internal-nf-budget"
POLICY_ID = "ala-total-internal-joint-cache-miss-budget-v1"
COMPLETENESS_INITIAL_OR_REPARAM_COMMITTED = (
    "initial_or_reparam_committed"
)
COMPLETENESS_ALL_RECORDED_STATES = "all_recorded_states"
COMPLETENESS_KEYED_MASK = "keyed_mask"
VALID_COMPLETENESS_POLICIES = (
    COMPLETENESS_INITIAL_OR_REPARAM_COMMITTED,
    COMPLETENESS_ALL_RECORDED_STATES,
    COMPLETENESS_KEYED_MASK,
)


class TotalInternalNFBudgetSafetyError(RuntimeError):
    """Fail-closed error for an inconsistent total-NF ledger."""


def validate_total_internal_nf_budget(value: Any) -> int:
    try:
        budget = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("total internal NF budget must be an integer") from error
    if str(budget) != str(value).strip() or budget < 1:
        raise ValueError("total internal NF budget must be a positive exact integer")
    return budget


def validate_completeness_policy(value: Any) -> str:
    policy = str(value)
    if policy not in VALID_COMPLETENESS_POLICIES:
        raise ValueError(
            "completeness policy must be initial_or_reparam_committed "
            "all_recorded_states, or keyed_mask"
        )
    return policy


def extract_total_internal_nf_budget(
    argv: Sequence[str],
) -> Tuple[int, List[str]]:
    """Remove exactly one shim-only total-NF option from an argv vector."""

    values: List[str] = []
    cleaned: List[str] = []
    tokens = list(argv)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == TOTAL_INTERNAL_NF_FLAG:
            if index + 1 >= len(tokens):
                raise ValueError(f"{TOTAL_INTERNAL_NF_FLAG} requires one value")
            values.append(tokens[index + 1])
            index += 2
            continue
        prefix = TOTAL_INTERNAL_NF_FLAG + "="
        if token.startswith(prefix):
            values.append(token[len(prefix) :])
            index += 1
            continue
        cleaned.append(token)
        index += 1
    if len(values) != 1:
        raise ValueError(
            f"exactly one {TOTAL_INTERNAL_NF_FLAG} option is required"
        )
    return validate_total_internal_nf_budget(values[0]), cleaned


@dataclass
class InstalledTotalInternalNFBudget:
    """One-process limiter plus result-persistence audit."""

    ala_core_module: Any
    budget: int
    original_query: Any
    completeness_policy: str
    completeness_mask_key: Optional[str] = None
    successful_internal_joint_evaluations: int = 0
    blocked_internal_misses: int = 0
    blocked_image_indices: List[int] = field(default_factory=list)
    full_band_preflight_calls: int = 0
    full_band_preflight_crops: int = 0
    full_band_preflight_zero_step_stops: int = 0
    complete_band_budget_stop_count: int = 0
    runner_module: Any = None
    original_runner_atomic_save_npz: Any = None
    runner_atomic_wrapper: Any = None
    restored: bool = False

    def is_internal_image(self, image_index: int) -> bool:
        return 0 < int(image_index) <= int(self.ala_core_module.N_INTERNAL)

    def query(self, evaluator: Any, positions_flat: Any, property_name: str = "force"):
        """Delegate one query while enforcing the shared successful-miss cap."""

        if property_name not in {"energy", "force", "both"}:
            return self.original_query(evaluator, positions_flat, property_name)

        is_internal = self.is_internal_image(evaluator.image_index)
        cached = evaluator.has_cached(positions_flat)
        if (
            is_internal
            and not cached
            and self.successful_internal_joint_evaluations >= self.budget
        ):
            # Reuse ImageEvaluator's own blocked-miss path so its request and
            # outcome identities remain exact.  The temporary image-local cap
            # is restored immediately; this shared limiter remains authoritative.
            prior_limit = evaluator.max_joint_evaluations
            evaluator.max_joint_evaluations = int(
                evaluator.counters.joint_evaluations
            )
            try:
                self.original_query(evaluator, positions_flat, property_name)
            except self.ala_core_module.EvaluationBudgetReached:
                self.blocked_internal_misses += 1
                self.blocked_image_indices.append(int(evaluator.image_index))
                raise
            finally:
                evaluator.max_joint_evaluations = prior_limit
            raise TotalInternalNFBudgetSafetyError(
                "ImageEvaluator failed to block a total-budget cache miss"
            )

        before = int(evaluator.counters.joint_evaluations)
        result = self.original_query(evaluator, positions_flat, property_name)
        after = int(evaluator.counters.joint_evaluations)
        delta = after - before
        if delta not in (0, 1):
            raise TotalInternalNFBudgetSafetyError(
                "one ImageEvaluator query changed the joint counter by more than one"
            )
        if is_internal:
            self.successful_internal_joint_evaluations += delta
            if self.successful_internal_joint_evaluations > self.budget:
                raise TotalInternalNFBudgetSafetyError(
                    "total internal NF budget overshot"
                )
        return result

    def result_metadata(self, arrays: Any) -> dict:
        """Validate the runner ledger and return persistence-only metadata."""

        required = {
            "counter_vectors",
            "reparam_committed",
            "total_internal_joint_evaluations",
        }
        missing = sorted(required.difference(arrays))
        if missing:
            raise TotalInternalNFBudgetSafetyError(
                "total-budget result is missing: " + ", ".join(missing)
            )
        cost = np.asarray(
            arrays["total_internal_joint_evaluations"], dtype=np.int64
        )
        counters = np.asarray(arrays["counter_vectors"], dtype=np.int64)
        committed = np.asarray(arrays["reparam_committed"], dtype=bool)
        if cost.ndim != 1 or cost.size < 1:
            raise TotalInternalNFBudgetSafetyError(
                "total-budget cost trajectory must be nonempty and one-dimensional"
            )
        if counters.ndim != 2 or counters.shape[0] != cost.size:
            raise TotalInternalNFBudgetSafetyError(
                "total-budget counter vectors are misaligned"
            )
        if counters.shape[1] != int(self.ala_core_module.N_IMAGES):
            raise TotalInternalNFBudgetSafetyError(
                "total-budget counter vectors have the wrong image count"
            )
        if committed.shape != cost.shape:
            raise TotalInternalNFBudgetSafetyError(
                "total-budget completeness flags are misaligned"
            )
        internal_sums = np.sum(counters[:, 1:-1], axis=1, dtype=np.int64)
        if not np.array_equal(cost, internal_sums):
            raise TotalInternalNFBudgetSafetyError(
                "saved total NF differs from the internal counter-vector sum"
            )
        consumed = int(cost[-1])
        if consumed != int(self.successful_internal_joint_evaluations):
            raise TotalInternalNFBudgetSafetyError(
                "runtime total NF differs from the saved terminal ledger"
            )
        if consumed > self.budget or np.any(cost > self.budget):
            raise TotalInternalNFBudgetSafetyError(
                "saved total NF exceeds the declared budget"
            )

        if (
            self.completeness_policy
            == COMPLETENESS_INITIAL_OR_REPARAM_COMMITTED
        ):
            complete = committed.copy()
            complete[0] = True
        elif self.completeness_policy == COMPLETENESS_ALL_RECORDED_STATES:
            complete = np.ones(cost.shape, dtype=bool)
        elif self.completeness_policy == COMPLETENESS_KEYED_MASK:
            key = str(self.completeness_mask_key or "")
            if not key or key not in arrays:
                raise TotalInternalNFBudgetSafetyError(
                    "keyed completeness mask is missing from the result"
                )
            raw_complete = np.asarray(arrays[key])
            if raw_complete.dtype.kind != "b" or raw_complete.shape != cost.shape:
                raise TotalInternalNFBudgetSafetyError(
                    "keyed completeness mask must be boolean and cost-aligned"
                )
            complete = raw_complete.astype(bool, copy=True)
        else:  # pragma: no cover - install-time validation is authoritative.
            raise TotalInternalNFBudgetSafetyError(
                "unknown installed completeness policy"
            )
        complete_indices = np.flatnonzero(complete)
        if not complete_indices.size:
            raise TotalInternalNFBudgetSafetyError("initial state is not complete")
        if not bool(complete[0]):
            raise TotalInternalNFBudgetSafetyError(
                "keyed completeness mask must include the initial state"
            )
        last_complete_index = int(complete_indices[-1])
        last_complete_nf = int(cost[last_complete_index])
        terminal_complete = bool(complete[-1])
        terminal_tail_nf = consumed - last_complete_nf
        if terminal_tail_nf < 0:
            raise TotalInternalNFBudgetSafetyError(
                "last complete state lies beyond the terminal NF"
            )
        partial_tail = bool(not terminal_complete and terminal_tail_nf > 0)
        remaining = int(self.budget) - consumed
        if consumed == self.budget:
            budget_stop_reason = "total_internal_joint_budget_exhausted"
        elif (
            self.complete_band_budget_stop_count > 0
            and remaining < int(self.ala_core_module.N_INTERNAL)
        ):
            budget_stop_reason = (
                "total_internal_joint_budget_insufficient_for_full_band"
            )
        else:
            budget_stop_reason = (
                "runner_stopped_before_total_internal_joint_budget"
            )
        return {
            "total_internal_nf_budget_policy_id": np.array(POLICY_ID),
            "total_internal_nf_budget": np.array(self.budget, dtype=np.int64),
            "total_internal_nf_consumed": np.array(consumed, dtype=np.int64),
            "total_internal_nf_remaining": np.array(
                remaining, dtype=np.int64
            ),
            "total_internal_nf_budget_stop_reason": np.array(
                budget_stop_reason
            ),
            "total_internal_nf_budget_exhausted": np.array(
                consumed == self.budget
            ),
            "total_internal_nf_budget_no_overshoot": np.array(True),
            "total_internal_nf_endpoint_evaluations_excluded": np.array(True),
            "total_internal_nf_budget_blocked_miss_count": np.array(
                self.blocked_internal_misses, dtype=np.int64
            ),
            "total_internal_nf_budget_blocked_image_indices": np.asarray(
                self.blocked_image_indices, dtype=np.int64
            ),
            "total_internal_nf_budget_exhaustion_observed_at_cache_miss": (
                np.array(self.blocked_internal_misses > 0)
            ),
            "total_internal_nf_full_band_preflight_calls": np.array(
                self.full_band_preflight_calls, dtype=np.int64
            ),
            "total_internal_nf_full_band_preflight_crops": np.array(
                self.full_band_preflight_crops, dtype=np.int64
            ),
            "total_internal_nf_full_band_preflight_zero_step_stops": np.array(
                self.full_band_preflight_zero_step_stops, dtype=np.int64
            ),
            "total_internal_nf_complete_band_budget_stop_count": np.array(
                self.complete_band_budget_stop_count, dtype=np.int64
            ),
            "total_internal_nf_complete_state_definition": np.array(
                self.completeness_policy
            ),
            "total_internal_nf_complete_state_mask_key": np.array(
                self.completeness_mask_key or ""
            ),
            "total_internal_nf_last_complete_state_index": np.array(
                last_complete_index, dtype=np.int64
            ),
            "total_internal_nf_last_complete_state_nf": np.array(
                last_complete_nf, dtype=np.int64
            ),
            "total_internal_nf_terminal_record_is_complete_outer_state": (
                np.array(terminal_complete)
            ),
            "total_internal_nf_terminal_uncommitted_charged_tail_nf": np.array(
                terminal_tail_nf, dtype=np.int64
            ),
            "total_internal_nf_terminal_partial_tail_present": np.array(
                partial_tail
            ),
            "total_internal_nf_terminal_partial_tail_quarantined": np.array(
                partial_tail
            ),
        }

    def attach_runner(self, runner_module: Any) -> None:
        """Persist the ledger metadata outside the frozen production source."""

        if self.runner_module is not None:
            raise RuntimeError("total internal NF budget already has a runner")
        self.runner_module = runner_module
        self.original_runner_atomic_save_npz = runner_module.atomic_save_npz

        def atomic_save_with_total_budget(destination: Path, **arrays: Any) -> None:
            additions = self.result_metadata(arrays)
            collisions = sorted(set(additions).intersection(arrays))
            if collisions:
                raise TotalInternalNFBudgetSafetyError(
                    "total-budget metadata key collision: "
                    + ", ".join(collisions)
                )
            self.original_runner_atomic_save_npz(
                destination, **dict(arrays, **additions)
            )

        self.runner_atomic_wrapper = atomic_save_with_total_budget
        runner_module.atomic_save_npz = atomic_save_with_total_budget

    def restore(self) -> None:
        if self.restored:
            return
        image_evaluator = self.ala_core_module.ImageEvaluator
        if image_evaluator.query is self.query_wrapper:
            image_evaluator.query = self.original_query
        if (
            self.runner_module is not None
            and self.runner_module.atomic_save_npz is self.runner_atomic_wrapper
        ):
            self.runner_module.atomic_save_npz = self.original_runner_atomic_save_npz
        self.restored = True

    # Assigned by the installer because a bound method object is not stable
    # under repeated attribute access and therefore cannot be identity-tested.
    query_wrapper: Any = None


def install_total_internal_nf_budget(
    ala_core_module: Any,
    budget: int,
    completeness_policy: str = COMPLETENESS_INITIAL_OR_REPARAM_COMMITTED,
    completeness_mask_key: Optional[str] = None,
) -> InstalledTotalInternalNFBudget:
    """Patch one imported ``ala_core.ImageEvaluator`` class for this process."""

    budget = validate_total_internal_nf_budget(budget)
    completeness_policy = validate_completeness_policy(completeness_policy)
    if completeness_policy == COMPLETENESS_KEYED_MASK:
        completeness_mask_key = str(completeness_mask_key or "")
        if not completeness_mask_key:
            raise ValueError("keyed completeness policy requires a mask key")
    elif completeness_mask_key is not None:
        raise ValueError("completeness mask key requires keyed_mask policy")
    original_query = ala_core_module.ImageEvaluator.query
    installed = InstalledTotalInternalNFBudget(
        ala_core_module=ala_core_module,
        budget=budget,
        original_query=original_query,
        completeness_policy=completeness_policy,
        completeness_mask_key=completeness_mask_key,
    )

    def query_with_total_budget(
        evaluator: Any, positions_flat: Any, property_name: str = "force"
    ):
        return installed.query(evaluator, positions_flat, property_name)

    installed.query_wrapper = query_with_total_budget
    ala_core_module.ImageEvaluator.query = query_with_total_budget
    return installed
