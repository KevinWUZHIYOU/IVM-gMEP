#!/usr/bin/env python3
"""Runtime-only selector policies for the frozen FFrbk discrete solver.

The frozen discrete optimizer, production runner, and selected-step eta shim
remain unchanged on disk.  This module changes only which of the already
constructed and already screened FFrbk-H0 / FFrbk-initialized L-BFGS
proposals is selected:

``one_way_takeover``
    Start every fixed-path solve with the original capped-decrease selector.
    The first strict L-BFGS win latches that *one* frozen subproblem's
    preference to L-BFGS.  Later valid L-BFGS proposals are selected
    unconditionally; an invalid L-BFGS proposal still fails over to valid H0
    without clearing the preference latch.

``forced_operator_lbfgs_selection``
    Select every valid FFrbk-initialized L-BFGS proposal, including the exact
    H0/L-BFGS tie at empty history.  The unchanged frozen loop still constructs
    its zero-physical-NF H0 shadow proposal, so this is deliberately named a
    *selection* policy rather than an L-BFGS-only construction policy.  An
    invalid L-BFGS proposal fails over to that valid H0 shadow.

The existing eta shim is composed *after* this selector, so eta=1.1 is still
applied to the selected step and the dynamic atom-displacement cap is then
reapplied.  This selector performs no energy query, line search, backtrack,
or reparameterization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import inspect
import math
from pathlib import Path
import sys
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

import run_ivm_v9_discrete_selected_eta_shim as selected_eta


MODE_FLAG = "--discrete-ffrbk-selector-mode"
VALID_MODES = ("one_way_takeover", "forced_operator_lbfgs_selection")
FIXED_ALPHA = 15.0
FIXED_ETA = 1.1
FIXED_HISTORY = 7
FIXED_INNER_MAX_STEPS = 10
FIXED_STOP_MODE = "r08c10"
FIXED_RESIDUAL_RATIO = 0.8
FIXED_KKT_ATOL = 0.0
FIXED_MAX_STEP = 1.0
RUNTIME_POLICY_ID = "ala-discrete-ffrbk-selector-runtime-v1"
MODE_POLICY_IDS = {
    "one_way_takeover": (
        "one-way-strict-lbfgs-takeover-per-fixed-subproblem-v1"
    ),
    "forced_operator_lbfgs_selection": (
        "forced-operator-lbfgs-selection-with-h0-shadow-v1"
    ),
}
OPERATOR_POLICY_ID = "frozen-ffrbk-h0-and-operator-initialized-lbfgs-v1"


class FFrbkSelectorSafetyError(RuntimeError):
    """Fail-closed error for the isolated selector runtime."""


def validate_mode(value: Any) -> str:
    mode = str(value)
    if mode not in VALID_MODES:
        raise ValueError(
            "FFrbk selector mode must be one_way_takeover or "
            "forced_operator_lbfgs_selection"
        )
    return mode


def validate_fixed_eta(value: Any) -> float:
    eta = selected_eta.validate_eta(value)
    if eta != FIXED_ETA:
        raise ValueError("FFrbk selector runtime requires fixed eta=1.1")
    return eta


def _extract_exactly_one_option(
    argv: Sequence[str], flag: str
) -> Tuple[str, List[str]]:
    values: List[str] = []
    cleaned: List[str] = []
    argv = list(argv)
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == flag:
            if index + 1 >= len(argv):
                raise ValueError(f"{flag} requires one value")
            values.append(argv[index + 1])
            index += 2
            continue
        prefix = flag + "="
        if token.startswith(prefix):
            values.append(token[len(prefix) :])
            index += 1
            continue
        cleaned.append(token)
        index += 1
    if len(values) != 1:
        raise ValueError(f"exactly one {flag} option is required")
    return values[0], cleaned


def extract_runtime_arguments(
    argv: Sequence[str],
) -> Tuple[str, float, List[str]]:
    """Remove this runtime's mode and the existing eta shim's option."""
    eta, without_eta = selected_eta.extract_eta_argument(argv)
    mode_text, cleaned = _extract_exactly_one_option(without_eta, MODE_FLAG)
    return validate_mode(mode_text), validate_fixed_eta(eta), cleaned


def _option_values(argv: Sequence[str], flag: str) -> List[str]:
    values: List[str] = []
    argv = list(argv)
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == flag:
            if index + 1 >= len(argv):
                raise ValueError(f"{flag} requires one value")
            values.append(argv[index + 1])
            index += 2
            continue
        prefix = flag + "="
        if token.startswith(prefix):
            values.append(token[len(prefix) :])
        index += 1
    return values


def _required_option(argv: Sequence[str], flag: str) -> str:
    values = _option_values(argv, flag)
    if len(values) != 1:
        raise ValueError(f"runtime requires exactly one {flag} option")
    return values[0]


def _require_switch(argv: Sequence[str], flag: str) -> None:
    if list(argv).count(flag) != 1:
        raise ValueError(f"runtime requires exactly one {flag} switch")


def _require_empty_or_absent(argv: Sequence[str], flag: str) -> None:
    values = _option_values(argv, flag)
    if len(values) > 1 or (values and values[0] != ""):
        raise ValueError(f"runtime forbids a nonempty {flag}")


def validate_runner_argv(argv: Sequence[str]) -> None:
    """Fail before OpenMM setup if a matched scientific control drifts."""
    exact_text = {
        "--solver": "state_ivm",
        "--frozen-subproblem": "discrete_b040",
        "--alpha-mode": "fixed",
        "--discrete-stop-mode": FIXED_STOP_MODE,
        "--reparam-mode": "state_curve",
        "--state-curve-direct-objective-acceptance": "per_image",
        "--state-globalization": "iperp_monotone",
    }
    for flag, expected in exact_text.items():
        if _required_option(argv, flag) != expected:
            raise ValueError(f"runtime requires {flag}={expected}")
    exact_float = {
        "--alpha": FIXED_ALPHA,
        "--max-step": FIXED_MAX_STEP,
        "--discrete-residual-ratio": FIXED_RESIDUAL_RATIO,
        "--plbfgs-kkt-atol": FIXED_KKT_ATOL,
        "--state-trust-fraction": 2.0,
        "--stop-complete-iperp": -1.0,
    }
    for flag, expected in exact_float.items():
        value = float(_required_option(argv, flag))
        if not math.isfinite(value) or value != expected:
            raise ValueError(f"runtime requires {flag}={expected:g}")
    exact_int = {
        "--history": FIXED_HISTORY,
        "--discrete-history": FIXED_HISTORY,
        "--discrete-inner-max-steps": FIXED_INNER_MAX_STEPS,
    }
    for flag, expected in exact_int.items():
        text = _required_option(argv, flag)
        try:
            value = int(text)
        except ValueError as error:
            raise ValueError(f"runtime requires integer {flag}") from error
        if str(value) != text.strip() or value != expected:
            raise ValueError(f"runtime requires {flag}={expected}")
    _require_switch(argv, "--state-curve-direct-solve")
    for flag in (
        "--alpha-schedule",
        "--alpha-precision-caps",
        "--alpha-relative-iperp-caps",
        "--max-step-schedule",
    ):
        _require_empty_or_absent(argv, flag)


def selector_implementation_id(base_implementation_id: str, mode: str) -> str:
    base = str(base_implementation_id)
    if not base:
        raise ValueError("base implementation ID must be nonempty")
    mode = validate_mode(mode)
    return "+".join(
        (
            base,
            RUNTIME_POLICY_ID,
            MODE_POLICY_IDS[mode],
            OPERATOR_POLICY_ID,
            "alpha15-m7-r08c10-kkt0",
        )
    )


def final_implementation_id(base_implementation_id: str, mode: str) -> str:
    """Return the exact selector-plus-existing-eta composition ID."""
    return selected_eta.shim_implementation_id(
        selector_implementation_id(base_implementation_id, mode), FIXED_ETA
    )


def _valid(proposal: Any) -> bool:
    try:
        return bool(proposal["valid"])
    except (KeyError, TypeError) as error:
        raise FFrbkSelectorSafetyError("proposal lacks valid flag") from error


def _decrease(proposal: Any) -> float:
    try:
        value = float(proposal["capped_decrease"])
    except (KeyError, TypeError, ValueError) as error:
        raise FFrbkSelectorSafetyError(
            "proposal lacks capped predicted decrease"
        ) from error
    if not math.isfinite(value):
        raise FFrbkSelectorSafetyError(
            "proposal capped predicted decrease is non-finite"
        )
    return value


def _same_valid_step(h0_proposal: Any, lbfgs_proposal: Any) -> bool:
    if not (_valid(h0_proposal) and _valid(lbfgs_proposal)):
        return False
    try:
        return bool(
            np.array_equal(
                np.asarray(h0_proposal["step"], dtype=np.float64),
                np.asarray(lbfgs_proposal["step"], dtype=np.float64),
            )
            and _decrease(h0_proposal) == _decrease(lbfgs_proposal)
        )
    except (KeyError, TypeError, ValueError):
        return False


@dataclass
class SelectorState:
    latched: bool = False
    selector_calls: int = 0


@dataclass(frozen=True)
class SelectorEvent:
    solver_call: int
    selector_call: int
    selection: str
    latched_before: bool
    latched_after: bool
    strict_win_trigger: bool
    forced_lbfgs: bool
    empty_history_tie: bool
    h0_failover: bool
    h0_valid: bool
    lbfgs_valid: bool
    h0_capped_decrease: float
    lbfgs_capped_decrease: float


def choose_direction(
    mode: str,
    state: SelectorState,
    h0_proposal: Any,
    lbfgs_proposal: Any,
    original_selector: Any,
    *,
    solver_call: int = 0,
) -> Tuple[Any, SelectorEvent]:
    """Apply one registered policy without inspecting energy/objective data."""
    mode = validate_mode(mode)
    h0_valid = _valid(h0_proposal)
    lbfgs_valid = _valid(lbfgs_proposal)
    h0_decrease = _decrease(h0_proposal)
    lbfgs_decrease = _decrease(lbfgs_proposal)
    latched_before = bool(state.latched)
    strict_win = False
    forced_lbfgs = False
    h0_failover = False
    empty_tie = _same_valid_step(h0_proposal, lbfgs_proposal)

    if mode == "one_way_takeover":
        if state.latched:
            if lbfgs_valid:
                selected = lbfgs_proposal
                forced_lbfgs = True
            elif h0_valid:
                selected = h0_proposal
                h0_failover = True
            else:
                selected = original_selector(h0_proposal, lbfgs_proposal)
        else:
            selected = original_selector(h0_proposal, lbfgs_proposal)
            strict_win = bool(
                h0_valid
                and lbfgs_valid
                and lbfgs_decrease > h0_decrease
            )
            if strict_win:
                if selected is not lbfgs_proposal:
                    raise FFrbkSelectorSafetyError(
                        "strict L-BFGS win disagrees with original selector"
                    )
                state.latched = True
    else:
        if lbfgs_valid:
            selected = lbfgs_proposal
            # The policy, rather than capped-decrease competition, owns this
            # choice.  This includes the exact empty-history tie.
            forced_lbfgs = True
        elif h0_valid:
            selected = h0_proposal
            h0_failover = True
        else:
            selected = original_selector(h0_proposal, lbfgs_proposal)

    try:
        selection = str(selected["label"])
    except (KeyError, TypeError) as error:
        raise FFrbkSelectorSafetyError("selected proposal lacks label") from error
    if selection not in ("H0", "LBFGS"):
        raise FFrbkSelectorSafetyError("unexpected selected proposal label")
    if not _valid(selected):
        raise FFrbkSelectorSafetyError("selector returned an invalid proposal")
    state.selector_calls += 1
    event = SelectorEvent(
        solver_call=int(solver_call),
        selector_call=state.selector_calls - 1,
        selection=selection,
        latched_before=latched_before,
        latched_after=bool(state.latched),
        strict_win_trigger=strict_win,
        forced_lbfgs=forced_lbfgs,
        empty_history_tie=empty_tie,
        h0_failover=h0_failover,
        h0_valid=h0_valid,
        lbfgs_valid=lbfgs_valid,
        h0_capped_decrease=h0_decrease,
        lbfgs_capped_decrease=lbfgs_decrease,
    )
    return selected, event


@dataclass
class SolverContext:
    solver_call: int
    state: SelectorState = field(default_factory=SelectorState)
    events: List[SelectorEvent] = field(default_factory=list)


@dataclass
class SelectorAudit:
    committed: List[SelectorEvent] = field(default_factory=list)
    uncommitted: List[SelectorEvent] = field(default_factory=list)
    aborted: List[SelectorEvent] = field(default_factory=list)
    contexts: List[SolverContext] = field(default_factory=list)
    fixed_solver_calls: int = 0
    latch_reset_count: int = 0
    aborted_solver_calls: int = 0
    apply_h0_inverse_calls: int = 0
    apply_h0_inverse_preconditioner_none_calls: int = 0
    joint_lbfgs_direction_calls: int = 0


def _validate_fixed_solver_arguments(arguments: Any) -> None:
    exact = {
        "alpha": FIXED_ALPHA,
        "inner_max_steps": FIXED_INNER_MAX_STEPS,
        "stop_mode": FIXED_STOP_MODE,
        "relative_residual": FIXED_RESIDUAL_RATIO,
        "history_size": FIXED_HISTORY,
        "kkt_atol": FIXED_KKT_ATOL,
    }
    for name, expected in exact.items():
        if arguments[name] != expected:
            raise ValueError(
                f"FFrbk selector runtime requires {name}={expected}"
            )
    # The CLI cap is fixed at one, but the unchanged production runner passes
    # its current dynamic active-step cap into each frozen subproblem.
    max_step = float(arguments["max_step"])
    if not math.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("fixed solver dynamic max_step must be positive")
    if arguments["preconditioner_bank"] is None:
        raise FFrbkSelectorSafetyError(
            "FFrbk selector runtime requires the frozen preconditioner bank"
        )


def _event_array(events: Sequence[SelectorEvent], field_name: str, dtype: Any):
    return np.asarray([getattr(event, field_name) for event in events], dtype=dtype)


@dataclass
class InstalledFFrbkSelectorShim:
    mode: str
    base_implementation_id: str
    pre_eta_implementation_id: str
    frozen_module: Any
    original_select_direction: Any
    original_solve: Any
    original_apply_h0_inverse: Any
    original_joint_lbfgs_direction: Any
    audit: SelectorAudit
    context_stack: List[SolverContext]
    final_implementation_id_value: Optional[str] = None
    runner_module: Any = None
    original_runner_atomic_save_npz: Any = None
    runner_atomic_wrapper: Any = None
    restored: bool = False

    @property
    def implementation_id(self) -> str:
        return (
            self.final_implementation_id_value
            if self.final_implementation_id_value is not None
            else self.pre_eta_implementation_id
        )

    def attach_runner(self, runner_module: Any, final_id: str) -> None:
        """Persist selector audit after the existing eta shim is attached."""
        if self.runner_module is not None:
            raise RuntimeError("FFrbk selector shim already has a runner")
        expected_final = final_implementation_id(
            self.base_implementation_id, self.mode
        )
        if str(final_id) != expected_final:
            raise FFrbkSelectorSafetyError("selector/eta implementation ID drift")
        if str(runner_module.DISCRETE_B040_IMPLEMENTATION_ID) != expected_final:
            raise FFrbkSelectorSafetyError("runner implementation ID drift")
        self.final_implementation_id_value = expected_final
        self.runner_module = runner_module
        self.original_runner_atomic_save_npz = runner_module.atomic_save_npz

        def atomic_save_with_selector(destination: Path, **arrays: Any) -> None:
            if str(np.asarray(arrays.get("frozen_subproblem", "")).item()) != (
                "discrete_b040"
            ):
                raise FFrbkSelectorSafetyError(
                    "selector runtime requires frozen_subproblem=discrete_b040"
                )
            if str(
                np.asarray(arrays.get("discrete_implementation_id", "")).item()
            ) != expected_final:
                raise FFrbkSelectorSafetyError(
                    "selector runtime result implementation ID drift"
                )
            fixed_scalars = {
                "alpha": FIXED_ALPHA,
                "alpha_mode": "fixed",
                "discrete_inner_max_steps": FIXED_INNER_MAX_STEPS,
                "discrete_stop_mode": FIXED_STOP_MODE,
                "discrete_residual_ratio": FIXED_RESIDUAL_RATIO,
                "discrete_history": FIXED_HISTORY,
                "plbfgs_kkt_atol": FIXED_KKT_ATOL,
                "reparam_mode": "state_curve",
                "state_curve_direct_objective_acceptance": "per_image",
            }
            for key, expected in fixed_scalars.items():
                if key not in arrays or np.asarray(arrays[key]).item() != expected:
                    raise FFrbkSelectorSafetyError(
                        f"selector runtime result drift at {key}"
                    )
            active_alpha = np.asarray(arrays.get("active_alpha", []), dtype=float)
            if not active_alpha.size or not np.all(active_alpha == FIXED_ALPHA):
                raise FFrbkSelectorSafetyError(
                    "selector runtime alpha was not fixed throughout"
                )
            selections = np.asarray(
                arrays.get("discrete_audit_selection", []), dtype="U8"
            )
            events = list(self.audit.committed)
            event_selections = _event_array(events, "selection", "U8")
            if not np.array_equal(selections, event_selections):
                raise FFrbkSelectorSafetyError(
                    "selector audit differs from committed runner selections"
                )
            if self.audit.latch_reset_count != self.audit.fixed_solver_calls:
                raise FFrbkSelectorSafetyError(
                    "selector latch was not reset once per fixed solver"
                )
            if self.audit.apply_h0_inverse_preconditioner_none_calls != 0:
                raise FFrbkSelectorSafetyError(
                    "a fixed solver attempted FFrbk without its bank"
                )
            if events and self.audit.apply_h0_inverse_calls <= 0:
                raise FFrbkSelectorSafetyError(
                    "FFrbk operator was not applied by fixed solver"
                )
            if self.mode == "one_way_takeover":
                by_call = {}
                for event in events:
                    by_call.setdefault(event.solver_call, []).append(event)
                for call_events in by_call.values():
                    if sum(e.strict_win_trigger for e in call_events) > 1:
                        raise FFrbkSelectorSafetyError(
                            "one fixed solver triggered takeover more than once"
                        )
                    if call_events and call_events[0].latched_before:
                        raise FFrbkSelectorSafetyError(
                            "one-way latch leaked across fixed solvers"
                        )
            additions = {
                "discrete_ffrbk_selector_runtime_policy_id": np.array(
                    RUNTIME_POLICY_ID
                ),
                "discrete_ffrbk_selector_mode": np.array(self.mode),
                "discrete_ffrbk_selector_mode_policy_id": np.array(
                    MODE_POLICY_IDS[self.mode]
                ),
                "discrete_ffrbk_selector_operator_policy_id": np.array(
                    OPERATOR_POLICY_ID
                ),
                "discrete_ffrbk_selector_base_implementation_id": np.array(
                    self.base_implementation_id
                ),
                "discrete_ffrbk_selector_pre_eta_implementation_id": np.array(
                    self.pre_eta_implementation_id
                ),
                "discrete_ffrbk_selector_final_implementation_id": np.array(
                    expected_final
                ),
                "discrete_ffrbk_selector_alpha": np.array(FIXED_ALPHA),
                "discrete_ffrbk_selector_eta": np.array(FIXED_ETA),
                "discrete_ffrbk_selector_stop_mode": np.array(FIXED_STOP_MODE),
                "discrete_ffrbk_selector_residual_ratio": np.array(
                    FIXED_RESIDUAL_RATIO
                ),
                "discrete_ffrbk_selector_inner_max_steps": np.array(
                    FIXED_INNER_MAX_STEPS
                ),
                "discrete_ffrbk_selector_history": np.array(FIXED_HISTORY),
                "discrete_ffrbk_selector_kkt_atol": np.array(FIXED_KKT_ATOL),
                "discrete_ffrbk_selector_operator_preserved": np.array(True),
                "discrete_ffrbk_selector_selector_only_change": np.array(True),
                "discrete_ffrbk_selector_h0_shadow_constructed": np.array(True),
                "discrete_ffrbk_selector_true_lbfgs_only_construction": np.array(
                    False
                ),
                "discrete_ffrbk_selector_invalid_lbfgs_h0_fallback_enabled": (
                    np.array(True)
                ),
                "discrete_ffrbk_selector_one_way_latch_is_preference": np.array(
                    self.mode == "one_way_takeover"
                ),
                "discrete_ffrbk_selector_post_select_eta_composed": np.array(
                    True
                ),
                "discrete_ffrbk_selector_energy_backtracking": np.array(False),
                "discrete_ffrbk_selector_extra_physical_evaluations": np.array(
                    0
                ),
                "discrete_ffrbk_selector_reparameterization_inside_subproblem": (
                    np.array(False)
                ),
                "discrete_ffrbk_selector_fixed_solver_calls": np.array(
                    self.audit.fixed_solver_calls
                ),
                "discrete_ffrbk_selector_latch_reset_count": np.array(
                    self.audit.latch_reset_count
                ),
                "discrete_ffrbk_selector_aborted_solver_calls": np.array(
                    self.audit.aborted_solver_calls
                ),
                "discrete_ffrbk_selector_apply_h0_inverse_calls": np.array(
                    self.audit.apply_h0_inverse_calls
                ),
                "discrete_ffrbk_selector_apply_h0_inverse_preconditioner_none_calls": (
                    np.array(
                        self.audit.apply_h0_inverse_preconditioner_none_calls
                    )
                ),
                "discrete_ffrbk_selector_joint_lbfgs_direction_calls": np.array(
                    self.audit.joint_lbfgs_direction_calls
                ),
                "discrete_ffrbk_selector_committed_selection_count": np.array(
                    len(self.audit.committed)
                ),
                "discrete_ffrbk_selector_uncommitted_selection_count": np.array(
                    len(self.audit.uncommitted)
                ),
                "discrete_ffrbk_selector_aborted_selection_count": np.array(
                    len(self.audit.aborted)
                ),
                "discrete_ffrbk_selector_audit_selection": event_selections,
            }
            event_fields = {
                "solver_call": int,
                "selector_call": int,
                "latched_before": bool,
                "latched_after": bool,
                "strict_win_trigger": bool,
                "forced_lbfgs": bool,
                "empty_history_tie": bool,
                "h0_failover": bool,
                "h0_valid": bool,
                "lbfgs_valid": bool,
                "h0_capped_decrease": float,
                "lbfgs_capped_decrease": float,
            }
            for field_name, dtype in event_fields.items():
                additions[
                    "discrete_ffrbk_selector_audit_" + field_name
                ] = _event_array(events, field_name, dtype)
            collisions = sorted(set(additions).intersection(arrays))
            if collisions:
                raise FFrbkSelectorSafetyError(
                    "selector metadata key collision: " + ", ".join(collisions)
                )
            self.original_runner_atomic_save_npz(
                destination, **{**arrays, **additions}
            )

        self.runner_atomic_wrapper = atomic_save_with_selector
        runner_module.atomic_save_npz = atomic_save_with_selector

    def restore(self) -> None:
        """Restore after the composed eta shim has first been restored."""
        if self.restored:
            return
        if (
            self.runner_module is not None
            and self.runner_module.atomic_save_npz is self.runner_atomic_wrapper
        ):
            self.runner_module.atomic_save_npz = self.original_runner_atomic_save_npz
        if self.runner_module is not None:
            self.runner_module.DISCRETE_B040_IMPLEMENTATION_ID = (
                self.base_implementation_id
            )
            if hasattr(self.runner_module, "solve_frozen_discrete_b040_band"):
                self.runner_module.solve_frozen_discrete_b040_band = (
                    self.original_solve
                )
        self.frozen_module.select_direction = self.original_select_direction
        self.frozen_module.solve_frozen_discrete_b040_band = self.original_solve
        self.frozen_module.apply_block_h0_inverse = self.original_apply_h0_inverse
        self.frozen_module.joint_lbfgs_direction = (
            self.original_joint_lbfgs_direction
        )
        self.frozen_module.IMPLEMENTATION_ID = self.base_implementation_id
        self.restored = True


def install_ffrbk_selector_shim(
    frozen_module: Any, mode: str
) -> InstalledFFrbkSelectorShim:
    """Patch only selector choice plus read-only operator/audit wrappers."""
    mode = validate_mode(mode)
    base_id = str(frozen_module.IMPLEMENTATION_ID)
    pre_eta_id = selector_implementation_id(base_id, mode)
    original_select = frozen_module.select_direction
    original_solve = frozen_module.solve_frozen_discrete_b040_band
    original_apply = frozen_module.apply_block_h0_inverse
    original_joint = frozen_module.joint_lbfgs_direction
    original_signature = inspect.signature(original_solve)
    audit = SelectorAudit()
    context_stack: List[SolverContext] = []

    def select_with_policy(h0_proposal: Any, lbfgs_proposal: Any) -> Any:
        if not context_stack:
            raise FFrbkSelectorSafetyError(
                "selector invoked outside a registered fixed solver"
            )
        context = context_stack[-1]
        selected, event = choose_direction(
            mode,
            context.state,
            h0_proposal,
            lbfgs_proposal,
            original_select,
            solver_call=context.solver_call,
        )
        context.events.append(event)
        return selected

    def apply_with_operator_audit(
        covector: Any,
        tangents: Any,
        preconditioner_bank: Any,
        alpha: float,
    ) -> Any:
        if context_stack:
            audit.apply_h0_inverse_calls += 1
            if preconditioner_bank is None:
                audit.apply_h0_inverse_preconditioner_none_calls += 1
                raise FFrbkSelectorSafetyError(
                    "FFrbk apply received no preconditioner bank"
                )
        return original_apply(
            covector, tangents, preconditioner_bank, alpha
        )

    def joint_with_operator_audit(*args: Any, **kwargs: Any) -> Any:
        if context_stack:
            audit.joint_lbfgs_direction_calls += 1
        return original_joint(*args, **kwargs)

    def solve_with_fresh_latch(*args: Any, **kwargs: Any) -> Any:
        bound = original_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        _validate_fixed_solver_arguments(bound.arguments)
        audit.fixed_solver_calls += 1
        audit.latch_reset_count += 1
        context = SolverContext(solver_call=audit.fixed_solver_calls - 1)
        context_stack.append(context)
        try:
            result = original_solve(*bound.args, **bound.kwargs)
        except BaseException:
            audit.aborted_solver_calls += 1
            audit.aborted.extend(context.events)
            raise
        finally:
            popped = context_stack.pop()
            if popped is not context:
                raise FFrbkSelectorSafetyError(
                    "selector fixed-solver context stack corruption"
                )
        report = result[8]
        committed_count = int(report["joint_steps"])
        if committed_count < 0 or committed_count > len(context.events):
            raise FFrbkSelectorSafetyError(
                "selector cannot identify committed fixed-solver events"
            )
        committed = context.events[:committed_count]
        uncommitted = context.events[committed_count:]
        report_selection = np.asarray(
            report["direction_selection"], dtype="U8"
        )
        event_selection = _event_array(committed, "selection", "U8")
        if not np.array_equal(report_selection, event_selection):
            raise FFrbkSelectorSafetyError(
                "fixed-solver report disagrees with selector events"
            )
        audit.committed.extend(committed)
        audit.uncommitted.extend(uncommitted)
        audit.contexts.append(context)
        return result

    frozen_module.select_direction = select_with_policy
    frozen_module.apply_block_h0_inverse = apply_with_operator_audit
    frozen_module.joint_lbfgs_direction = joint_with_operator_audit
    frozen_module.solve_frozen_discrete_b040_band = solve_with_fresh_latch
    frozen_module.IMPLEMENTATION_ID = pre_eta_id
    return InstalledFFrbkSelectorShim(
        mode=mode,
        base_implementation_id=base_id,
        pre_eta_implementation_id=pre_eta_id,
        frozen_module=frozen_module,
        original_select_direction=original_select,
        original_solve=original_solve,
        original_apply_h0_inverse=original_apply,
        original_joint_lbfgs_direction=original_joint,
        audit=audit,
        context_stack=context_stack,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    process_argv = list(sys.argv if argv is None else argv)
    mode, eta, runner_argv = extract_runtime_arguments(process_argv)
    validate_runner_argv(runner_argv)
    sys.argv = runner_argv

    frozen = importlib.import_module("frozen_discrete_b040")
    selector_installed = install_ffrbk_selector_shim(frozen, mode)
    # Composition order is the scientific contract: choose under the new
    # policy first, then reuse the locked post-selection eta=1.1 shim.
    eta_installed = selected_eta.install_frozen_eta_shim(frozen, eta)
    expected_final = final_implementation_id(
        selector_installed.base_implementation_id, mode
    )
    if eta_installed.implementation_id != expected_final:
        raise FFrbkSelectorSafetyError("unexpected selector/eta composition ID")
    runner = importlib.import_module(
        "run_ivm_v9_spring_projected_production_20260730"
    )
    eta_installed.attach_runner(runner)
    selector_installed.attach_runner(runner, eta_installed.implementation_id)
    runner.main()


if __name__ == "__main__":
    main()
