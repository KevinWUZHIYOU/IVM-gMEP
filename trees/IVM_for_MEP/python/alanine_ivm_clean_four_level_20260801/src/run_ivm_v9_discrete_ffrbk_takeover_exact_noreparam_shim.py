#!/usr/bin/env python3
"""Exact-like, no-reparameterization runtime for the Ala alpha grid.

This entry point composes four already isolated pieces in one fresh process:

* the frozen direct-coordinate ``discrete_b040`` subproblem;
* the one-way FFrbk-H0/operator-L-BFGS selector;
* the locked post-selection ``eta=1.1`` multiplier; and
* a candidate-aware hard total-internal-NF limiter.

"Exact" is deliberately operational: every completed frozen-path problem
must reduce the joint natural residual to ``R/R0 <= 0.1``.  The positive
integer cap 5000 is an implementation safety guard, not a scientific stopping
rule; a completed solve that reaches it is a failed cell. Reparameterization
is disabled with ``--reparam-every 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

import run_ivm_v9_discrete_ffrbk_takeover_grid_shim as grid_runtime
import run_ivm_v9_discrete_ffrbk_takeover_selector_shim as fixed_selector
import run_ivm_v9_discrete_exact_candidate_budget_solver as candidate_budget
import run_ivm_v9_discrete_selected_eta_shim as selected_eta
import run_ivm_v9_total_internal_nf_budget as total_budget


POLICY_ID = "ala-discrete-exactlike-r01-no-reparam-alpha-grid-runtime-v1"
COMPLETE_BAND_PREFLIGHT_POLICY_ID = (
    "ala-exact-no-reparam-total-nf-complete-band-preflight-v1"
)
FAMILY = "exact_no_reparam"
OPERATIONAL_DEFINITION = (
    "exact-complete states are the initial state and published outer states "
    "following frozen solves with normalized R/R0<=0.1; paid trials from an "
    "interrupted solve are charged but excluded"
)
EXACT_COMPLETE_MASK_KEY = "exact_no_reparam_complete_state_mask"
RESIDUAL_RATIO = 0.1
INNER_MAX_STEPS = 5000
TOTAL_INTERNAL_NF_BUDGET = 5000
PER_IMAGE_SAFETY_BUDGET = 5000
PHYSICAL_MAX_STEP = 1.0
ETA = 1.1
HISTORY = 7
KKT_ATOL = 0.0
STOP_MODE = "r08c10"  # Historical API token; the scalar ratio is authoritative.


class ExactNoReparamSafetyError(RuntimeError):
    """Fail closed when the exact/no-reparameterization contract drifts."""


def _array_sha256(value: Any) -> str:
    contiguous = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


@dataclass
class ExactSolveEvent:
    stop_reason: str
    joint_steps: int
    global_relative_residual: float
    exact_endpoint_nf: int
    raw_exact_endpoint_path: Any = None
    published_exact_endpoint_path: Any = None


@dataclass
class InstalledExactSolveCompletionAudit:
    frozen_module: Any
    original_solve: Any
    events: list[ExactSolveEvent] = field(default_factory=list)
    initial_path: Any = None
    initial_nf: int = -1
    solve_wrapper: Any = None
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        if self.frozen_module.solve_frozen_discrete_b040_band is self.solve_wrapper:
            self.frozen_module.solve_frozen_discrete_b040_band = self.original_solve
        self.restored = True


def install_exact_solve_completion_audit(
    frozen_module: Any,
) -> InstalledExactSolveCompletionAudit:
    """Record only residual-converged frozen-solve endpoint NoFs as exact."""

    original = frozen_module.solve_frozen_discrete_b040_band
    signature = inspect.signature(original)
    installed = InstalledExactSolveCompletionAudit(
        frozen_module=frozen_module, original_solve=original
    )

    def solve_with_exact_completion_audit(*args: Any, **kwargs: Any):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        starting_path = np.asarray(bound.arguments["path"], dtype=np.float64)
        evaluators = bound.arguments["evaluators"]
        starting_nf = int(
            sum(
                evaluator.counters.joint_evaluations
                for evaluator in evaluators[1:-1]
            )
        )
        if installed.initial_path is None:
            installed.initial_path = starting_path.copy()
            installed.initial_nf = starting_nf
        if installed.events:
            previous = installed.events[-1]
            if (
                previous.exact_endpoint_nf >= 0
                and previous.published_exact_endpoint_path is None
            ):
                if starting_nf != previous.exact_endpoint_nf:
                    raise ExactNoReparamSafetyError(
                        "next frozen solve did not start at the prior exact NF"
                    )
                previous.published_exact_endpoint_path = starting_path.copy()
        result = original(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 9:
            raise ExactNoReparamSafetyError("unexpected frozen-solve result shape")
        synchronized_counters = list(result[7])
        report = result[8]
        stop_reason = str(report["stop_reason"])
        joint_steps = int(report["joint_steps"])
        relative = float(report["global_relative_residual"])
        if joint_steps != len(synchronized_counters):
            raise ExactNoReparamSafetyError(
                "frozen solve step/counter audit is misaligned"
            )
        exact_endpoint_nf = -1
        raw_exact_endpoint_path = None
        if stop_reason in (
            "evaluated-relative-natural-residual",
            "evaluated-absolute-natural-residual",
        ):
            if joint_steps < 1 or not synchronized_counters:
                raise ExactNoReparamSafetyError(
                    "relative-residual solve has no evaluated endpoint"
                )
            if not np.isfinite(relative) or relative > RESIDUAL_RATIO + 1.0e-12:
                raise ExactNoReparamSafetyError(
                    "relative-residual solve stopped above r=0.1"
                )
            counters = np.asarray(
                synchronized_counters[-1]["joint_evaluations"], dtype=np.int64
            )
            if counters.shape != (22,):
                raise ExactNoReparamSafetyError(
                    "frozen endpoint counter vector has the wrong shape"
                )
            exact_endpoint_nf = int(np.sum(counters[1:-1], dtype=np.int64))
            raw_exact_endpoint_path = np.asarray(
                result[4][-1], dtype=np.float64
            ).copy()
        installed.events.append(
            ExactSolveEvent(
                stop_reason=stop_reason,
                joint_steps=joint_steps,
                global_relative_residual=relative,
                exact_endpoint_nf=exact_endpoint_nf,
                raw_exact_endpoint_path=raw_exact_endpoint_path,
            )
        )
        return result

    installed.solve_wrapper = solve_with_exact_completion_audit
    frozen_module.solve_frozen_discrete_b040_band = solve_with_exact_completion_audit
    return installed


def restore_runner_candidate_identity(
    runner_module: Any,
    candidate_installed: candidate_budget.InstalledCandidateAwareBudgetSolver,
) -> None:
    """Restore aliases captured when the runner was imported under the shim."""

    runner_module.solve_frozen_discrete_b040_band = (
        candidate_installed.original_solve
    )
    runner_module.DISCRETE_B040_IMPLEMENTATION_ID = (
        candidate_installed.original_implementation_id
    )


def _required_option(argv: Sequence[str], flag: str) -> str:
    return grid_runtime._required_option(argv, flag)


def _require_empty_or_absent(argv: Sequence[str], flag: str) -> None:
    values = grid_runtime._option_values(argv, flag)
    if len(values) > 1 or (values and values[0] != ""):
        raise ValueError(f"exact/no-reparam runtime forbids a nonempty {flag}")


def validate_exact_no_reparam_runner_argv(
    selector_module: Any,
    runner_argv: Sequence[str],
    controls: grid_runtime.GridControls,
    mode: str,
    eta: float,
) -> None:
    """Validate the complete scientific command before OpenMM is imported."""

    controls = grid_runtime.validate_grid_controls(controls)
    if controls.residual_ratio != RESIDUAL_RATIO:
        raise ValueError("exact/no-reparam runtime requires residual ratio 0.1")
    if controls.inner_max_steps != INNER_MAX_STEPS:
        raise ValueError("exact/no-reparam runtime requires safety cap 5000")
    if controls.total_internal_nf_budget != TOTAL_INTERNAL_NF_BUDGET:
        raise ValueError("exact/no-reparam runtime requires total internal NF=5000")
    if mode != "one_way_takeover":
        raise ValueError("exact/no-reparam runtime requires one_way_takeover")
    if float(eta) != ETA:
        raise ValueError("exact/no-reparam runtime requires eta=1.1")

    # Reuse the mature selector validator after its constants have been
    # parameterized by ``configured_selector_policy``.  ``state_curve`` is an
    # inert retained token here; the separately checked frequency zero is the
    # actual no-reparameterization switch.
    selector_module.validate_runner_argv(runner_argv)
    required = {
        "--solver": "state_ivm",
        "--frozen-subproblem": "discrete_b040",
        "--alpha-mode": "fixed",
        "--max-step": "1",
        "--history": "7",
        "--discrete-history": "7",
        "--discrete-stop-mode": STOP_MODE,
        "--discrete-residual-ratio": "0.1",
        "--discrete-inner-max-steps": "5000",
        "--plbfgs-kkt-rtol": "0.1",
        "--plbfgs-kkt-atol": "0",
        "--reparam-mode": "state_curve",
        "--reparam-every": "0",
        "--state-curve-direct-objective-acceptance": "per_image",
        "--state-globalization": "iperp_monotone",
        "--stop-iperp": "-1",
        "--stop-complete-iperp": "-1",
        "--budget": str(PER_IMAGE_SAFETY_BUDGET),
    }
    for flag, expected in required.items():
        if _required_option(runner_argv, flag) != expected:
            raise ValueError(
                f"exact/no-reparam runtime requires {flag}={expected}"
            )
    if list(runner_argv).count("--state-curve-direct-solve") != 1:
        raise ValueError("exact/no-reparam runtime requires the retained direct switch")
    for flag in (
        "--alpha-schedule",
        "--alpha-precision-caps",
        "--alpha-relative-iperp-caps",
        "--max-step-schedule",
        "--reparam-every-schedule",
    ):
        _require_empty_or_absent(runner_argv, flag)


def _attach_exact_metadata(
    runner_module: Any,
    budget_installed: Any,
    completion_installed: InstalledExactSolveCompletionAudit,
    candidate_installed: candidate_budget.InstalledCandidateAwareBudgetSolver,
) -> tuple[Any, Any]:
    """Add a self-describing exact/no-reparam contract to the result NPZ."""

    original = runner_module.atomic_save_npz

    def atomic_save_with_exact_contract(destination: Path, **arrays: Any) -> None:
        expected = {
            "reparam_every": 0,
            "discrete_residual_ratio": RESIDUAL_RATIO,
            "discrete_inner_max_steps": INNER_MAX_STEPS,
            "plbfgs_kkt_atol": KKT_ATOL,
        }
        for key, value in expected.items():
            if key not in arrays or np.asarray(arrays[key]).item() != value:
                raise ExactNoReparamSafetyError(
                    f"exact/no-reparam result drift at {key}"
                )
        counters = np.asarray(arrays.get("counter_vectors", []), dtype=np.int64)
        cost = np.asarray(
            arrays.get("total_internal_joint_evaluations", []), dtype=np.int64
        )
        returned_states_evaluated = bool(
            np.asarray(
                arrays.get("discrete_all_returned_states_evaluated", False)
            ).item()
        )
        no_reparam = not (
            np.any(np.asarray(arrays.get("reparam_attempted", []), dtype=bool))
            or np.any(np.asarray(arrays.get("reparam_committed", []), dtype=bool))
        )
        complete_joint_bands = bool(
            counters.ndim == 2
            and counters.shape[1] == 22
            and counters.shape[0] > 0
            and cost.shape == (counters.shape[0],)
            and np.array_equal(
                cost, np.sum(counters[:, 1:-1], axis=1, dtype=np.int64)
            )
        )
        consumed = int(
            np.sum(counters[-1, 1:-1], dtype=np.int64)
            if counters.ndim == 2 and counters.shape[0] > 0
            else -1
        )
        preflight_guards_every_solve = bool(
            budget_installed.successful_internal_joint_evaluations == consumed
            and 0 < consumed <= TOTAL_INTERNAL_NF_BUDGET
            and budget_installed.full_band_preflight_calls > 0
        )
        budget_limited = bool(
            budget_installed.complete_band_budget_stop_count > 0
        )
        if budget_limited:
            preflight_guards_every_solve = bool(
                preflight_guards_every_solve
                and budget_installed.full_band_preflight_crops > 0
            )
        if not (
            returned_states_evaluated
            and no_reparam
            and complete_joint_bands
            and preflight_guards_every_solve
        ):
            raise ExactNoReparamSafetyError(
                "cannot certify every recorded no-reparam state as a complete "
                "joint band"
            )
        candidate_events = list(candidate_installed.events)
        candidate_stopped = np.asarray(
            [event.stopped_before_query for event in candidate_events], dtype=bool
        )
        if (
            len(candidate_events) != budget_installed.full_band_preflight_calls
            or int(np.count_nonzero(candidate_stopped))
            != budget_installed.full_band_preflight_crops
            or int(
                np.count_nonzero(
                    candidate_stopped
                    & (
                        np.asarray(
                            [event.joint_steps_before for event in candidate_events],
                            dtype=np.int64,
                        )
                        == 0
                    )
                )
            )
            != budget_installed.full_band_preflight_zero_step_stops
            or int(np.count_nonzero(candidate_stopped))
            != budget_installed.complete_band_budget_stop_count
        ):
            raise ExactNoReparamSafetyError(
                "candidate-aware complete-band preflight audit drifted"
            )
        saved_stop_reasons = np.asarray(
            arrays.get("discrete_audit_stop_reason", []), dtype="U64"
        )
        events = list(completion_installed.events)
        event_stop_reasons = np.asarray(
            [event.stop_reason for event in events], dtype="U64"
        )
        if not np.array_equal(saved_stop_reasons, event_stop_reasons):
            raise ExactNoReparamSafetyError(
                "runtime solve-completion audit differs from the runner"
            )
        stored_paths = np.asarray(arrays.get("stored_paths", []), dtype=np.float64)
        stored_calls = np.asarray(
            arrays.get("stored_path_calls", []), dtype=np.int64
        )
        history_calls = np.asarray(arrays.get("calls_per_image", []), dtype=np.int64)
        if (
            stored_paths.ndim != 3
            or stored_paths.shape[0] != cost.size
            or stored_calls.shape != cost.shape
            or history_calls.shape != cost.shape
            or not np.array_equal(stored_calls, history_calls)
        ):
            raise ExactNoReparamSafetyError(
                "save-all path history is not aligned with the cost trajectory"
            )
        endpoint_nf = np.asarray(
            [event.exact_endpoint_nf for event in events], dtype=np.int64
        )
        record_hashes = np.asarray(
            [_array_sha256(path) for path in stored_paths], dtype="U64"
        )
        if (
            completion_installed.initial_path is None
            or completion_installed.initial_nf < 0
        ):
            raise ExactNoReparamSafetyError(
                "exact completion audit did not capture the initial state"
            )
        exact_complete_mask = np.zeros(cost.shape, dtype=bool)
        initial_hash = _array_sha256(completion_installed.initial_path)
        initial_matches = (cost == completion_installed.initial_nf) & (
            record_hashes == initial_hash
        )
        if np.count_nonzero(initial_matches) < 1:
            raise ExactNoReparamSafetyError(
                "saved trajectory does not contain a path-identical initial state"
            )
        exact_complete_mask |= initial_matches

        event_aligned = np.zeros(len(events), dtype=bool)
        raw_endpoint_paths = np.full(
            (len(events),) + stored_paths.shape[1:], np.nan, dtype=np.float64
        )
        published_endpoint_paths = np.full_like(raw_endpoint_paths, np.nan)
        raw_endpoint_hashes = np.full(len(events), "", dtype="U64")
        published_endpoint_hashes = np.full(len(events), "", dtype="U64")
        for index, event in enumerate(events):
            if event.exact_endpoint_nf < 0:
                continue
            if event.raw_exact_endpoint_path is None:
                raise ExactNoReparamSafetyError(
                    "converged solve lacks its raw evaluated endpoint"
                )
            raw_endpoint_paths[index] = event.raw_exact_endpoint_path
            raw_endpoint_hashes[index] = _array_sha256(
                event.raw_exact_endpoint_path
            )
            if event.published_exact_endpoint_path is None:
                if index != len(events) - 1 or cost[-1] != event.exact_endpoint_nf:
                    raise ExactNoReparamSafetyError(
                        "cannot identify a converged solve's published endpoint"
                    )
                event.published_exact_endpoint_path = stored_paths[-1].copy()
            published = np.asarray(
                event.published_exact_endpoint_path, dtype=np.float64
            )
            published_endpoint_paths[index] = published
            published_hash = _array_sha256(published)
            published_endpoint_hashes[index] = published_hash
            matches = (cost == event.exact_endpoint_nf) & (
                record_hashes == published_hash
            )
            event_aligned[index] = bool(np.any(matches))
            exact_complete_mask |= matches

        exact_complete_indices = np.flatnonzero(exact_complete_mask)
        last_complete_nf = int(cost[exact_complete_indices[-1]])
        charged_noncomplete_mask = ~exact_complete_mask
        additions = {
            "exact_no_reparam_runtime_policy_id": np.array(POLICY_ID),
            "exact_no_reparam_family": np.array(FAMILY),
            "exact_no_reparam_operational_definition": np.array(
                OPERATIONAL_DEFINITION
            ),
            "exact_no_reparam_enabled": np.array(True),
            "exact_no_reparam_residual_ratio": np.array(RESIDUAL_RATIO),
            "exact_no_reparam_scientific_maxstep_unbounded_required": np.array(
                True
            ),
            "exact_no_reparam_implementation_safety_inner_cap": np.array(
                INNER_MAX_STEPS, dtype=np.int64
            ),
            "exact_no_reparam_inner_cap_required_nonbinding": np.array(True),
            "exact_no_reparam_completed_cap_stop_allowed": np.array(False),
            "exact_no_reparam_complete_state_definition": np.array(
                "initial_plus_published_outer_states_after_normalized_"
                "residual_converged_frozen_solves"
            ),
            EXACT_COMPLETE_MASK_KEY: exact_complete_mask,
            "exact_no_reparam_complete_state_indices": exact_complete_indices,
            "exact_no_reparam_complete_state_nf": cost[exact_complete_mask],
            "exact_no_reparam_charged_noncomplete_state_mask": (
                charged_noncomplete_mask
            ),
            "exact_no_reparam_all_recorded_states_complete": np.array(
                bool(np.all(exact_complete_mask))
            ),
            "exact_no_reparam_terminal_record_is_complete": np.array(
                bool(exact_complete_mask[-1])
            ),
            "exact_no_reparam_terminal_charged_tail_nf": np.array(
                consumed - last_complete_nf, dtype=np.int64
            ),
            "exact_no_reparam_solve_stop_reason": event_stop_reasons,
            "exact_no_reparam_solve_joint_steps": np.asarray(
                [event.joint_steps for event in events], dtype=np.int64
            ),
            "exact_no_reparam_solve_global_relative_residual": np.asarray(
                [event.global_relative_residual for event in events], dtype=float
            ),
            "exact_no_reparam_solve_exact_endpoint_nf": endpoint_nf,
            "exact_no_reparam_solve_raw_exact_endpoint_path": raw_endpoint_paths,
            "exact_no_reparam_solve_raw_exact_endpoint_path_sha256": (
                raw_endpoint_hashes
            ),
            "exact_no_reparam_solve_published_exact_endpoint_path": (
                published_endpoint_paths
            ),
            "exact_no_reparam_solve_published_exact_endpoint_path_sha256": (
                published_endpoint_hashes
            ),
            "exact_no_reparam_solve_exact_endpoint_aligned_to_saved_state": (
                event_aligned
            ),
            "exact_no_reparam_initial_path": np.asarray(
                completion_installed.initial_path, dtype=np.float64
            ),
            "exact_no_reparam_initial_path_sha256": np.array(initial_hash),
            "exact_no_reparam_initial_nf": np.array(
                completion_installed.initial_nf, dtype=np.int64
            ),
            "exact_no_reparam_complete_band_preflight": np.array(True),
            "exact_no_reparam_complete_band_preflight_policy_id": np.array(
                COMPLETE_BAND_PREFLIGHT_POLICY_ID
            ),
            "exact_no_reparam_candidate_budget_policy_id": np.array(
                candidate_budget.POLICY_ID
            ),
            "exact_no_reparam_candidate_required_internal_misses": np.asarray(
                [
                    event.required_internal_misses
                    for event in candidate_events
                ],
                dtype=np.int64,
            ),
            "exact_no_reparam_candidate_remaining_internal_nf_before": np.asarray(
                [
                    event.remaining_internal_nf_before
                    for event in candidate_events
                ],
                dtype=np.int64,
            ),
            "exact_no_reparam_candidate_stopped_before_query": candidate_stopped,
            "exact_no_reparam_candidate_joint_steps_before": np.asarray(
                [event.joint_steps_before for event in candidate_events],
                dtype=np.int64,
            ),
        }
        collisions = sorted(set(additions).intersection(arrays))
        if collisions:
            raise ExactNoReparamSafetyError(
                "exact/no-reparam metadata collision: " + ", ".join(collisions)
            )
        original(destination, **dict(arrays, **additions))

    runner_module.atomic_save_npz = atomic_save_with_exact_contract
    return original, atomic_save_with_exact_contract


def main(argv: Optional[Sequence[str]] = None) -> None:
    process_argv = list(sys.argv if argv is None else argv)
    controls, mode, eta, runner_argv = grid_runtime.extract_runtime_arguments(
        process_argv
    )
    old_argv = sys.argv
    budget_installed = None
    selector_installed = None
    eta_installed = None
    runner = None
    original_atomic = None
    exact_atomic = None
    candidate_installed = None
    completion_installed = None
    with grid_runtime.configured_selector_policy(
        fixed_selector, controls
    ) as selector_module:
        validate_exact_no_reparam_runner_argv(
            selector_module, runner_argv, controls, mode, eta
        )
        sys.argv = list(runner_argv)
        ala_core = importlib.import_module("ala_core")
        budget_installed = total_budget.install_total_internal_nf_budget(
            ala_core,
            controls.total_internal_nf_budget,
            completeness_policy=total_budget.COMPLETENESS_KEYED_MASK,
            completeness_mask_key=EXACT_COMPLETE_MASK_KEY,
        )
        try:
            frozen = importlib.import_module("frozen_discrete_b040")
            candidate_installed = (
                candidate_budget.install_candidate_aware_budget_solver(
                    frozen, budget_installed
                )
            )
            selector_installed = selector_module.install_ffrbk_selector_shim(
                frozen, mode
            )
            eta_installed = selected_eta.install_frozen_eta_shim(frozen, eta)
            expected_final = selector_module.final_implementation_id(
                selector_installed.base_implementation_id, mode
            )
            if eta_installed.implementation_id != expected_final:
                raise ExactNoReparamSafetyError(
                    "unexpected selector/eta implementation identity"
                )
            completion_installed = install_exact_solve_completion_audit(frozen)
            runner = importlib.import_module(
                "run_ivm_v9_spring_projected_production_20260730"
            )
            eta_installed.attach_runner(runner)
            selector_installed.attach_runner(
                runner, eta_installed.implementation_id
            )
            budget_installed.attach_runner(runner)
            original_atomic, exact_atomic = _attach_exact_metadata(
                runner,
                budget_installed,
                completion_installed,
                candidate_installed,
            )
            runner.main()
        finally:
            if (
                runner is not None
                and exact_atomic is not None
                and runner.atomic_save_npz is exact_atomic
            ):
                runner.atomic_save_npz = original_atomic
            if budget_installed is not None:
                budget_installed.restore()
            if completion_installed is not None:
                completion_installed.restore()
            if eta_installed is not None:
                eta_installed.restore()
            if selector_installed is not None:
                selector_installed.restore()
            if candidate_installed is not None:
                candidate_installed.restore()
                if runner is not None:
                    restore_runner_candidate_identity(
                        runner, candidate_installed
                    )
            sys.argv = old_argv


if __name__ == "__main__":
    main()
