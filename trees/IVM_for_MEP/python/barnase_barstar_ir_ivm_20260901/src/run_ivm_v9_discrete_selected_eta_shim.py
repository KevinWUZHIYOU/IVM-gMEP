#!/usr/bin/env python3
"""Runtime-only selected-step multiplier for frozen discrete B040.

This entry point deliberately leaves both the frozen optimizer and production
runner source files unchanged.  The original H0/L-BFGS proposals are screened
and selected first.  Only the already-selected step is then multiplied by the
registered positive scalar ``eta`` and passed through the original dynamic
whole-band cap once more.

The shim is intentionally unsuitable for legacy/non-discrete runs.  Its
implementation ID includes the exact binary64 value of ``eta``; the production
runner already includes that ID in its configuration signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
from pathlib import Path
import sys
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


ETA_FLAG = "--discrete-step-multiplier-eta"
SHIM_POLICY_ID = "post-select-selected-step-eta-recap-v1"
APPLICATION_STAGE = "after-original-select-before-trial"
PILOT_ETA_VALUES = (1.0, 1.25, 1.5, 2.0)


def validate_eta(value: Any) -> float:
    """Return one finite, strictly positive binary64 multiplier."""
    eta = float(value)
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("selected-step eta must be finite and strictly positive")
    return eta


def eta_token(eta: float) -> str:
    """Return an injective, implementation-ID-safe binary64 token."""
    eta = validate_eta(eta)
    return eta.hex().replace(".", "d").replace("+", "p").replace("-", "m")


def shim_implementation_id(base_implementation_id: str, eta: float) -> str:
    """Return the exact eta-specific ID used by the runner configuration hash."""
    base = str(base_implementation_id)
    if not base:
        raise ValueError("base implementation ID must be nonempty")
    return f"{base}+{SHIM_POLICY_ID}+eta-{eta_token(eta)}"


def extract_eta_argument(argv: Sequence[str]) -> Tuple[float, List[str]]:
    """Remove exactly one shim-only eta option from an argv vector.

    The remaining vector can be passed unchanged to the original runner's
    parser, which intentionally knows nothing about this runtime extension.
    """
    values: List[str] = []
    cleaned: List[str] = []
    index = 0
    argv = list(argv)
    while index < len(argv):
        token = argv[index]
        if token == ETA_FLAG:
            if index + 1 >= len(argv):
                raise ValueError(f"{ETA_FLAG} requires one value")
            values.append(argv[index + 1])
            index += 2
            continue
        prefix = ETA_FLAG + "="
        if token.startswith(prefix):
            values.append(token[len(prefix) :])
            index += 1
            continue
        cleaned.append(token)
        index += 1
    if len(values) != 1:
        raise ValueError(f"exactly one {ETA_FLAG} option is required")
    return validate_eta(values[0]), cleaned


def _max_atomic_displacement(step: Any) -> float:
    step = np.asarray(step, dtype=np.float64)
    value = float(np.max(np.linalg.norm(step.reshape(-1, 3), axis=1)))
    if not math.isfinite(value):
        raise RuntimeError("selected-step eta produced a non-finite displacement")
    return value


@dataclass(frozen=True)
class EtaEvent:
    label: str
    requested_multiplier: float
    applied_multiplier: float
    cap_hit: bool
    dynamic_max_step: float
    base_max_atomic_displacement: float
    requested_max_atomic_displacement: float
    applied_max_atomic_displacement: float


@dataclass
class EtaAudit:
    """Committed events plus fail-closed bookkeeping for solver-local trials."""

    committed: List[EtaEvent] = field(default_factory=list)
    uncommitted: List[EtaEvent] = field(default_factory=list)
    aborted: List[EtaEvent] = field(default_factory=list)
    standalone: List[EtaEvent] = field(default_factory=list)
    capture_stack: List[List[EtaEvent]] = field(default_factory=list)

    def record(self, event: EtaEvent) -> None:
        if self.capture_stack:
            self.capture_stack[-1].append(event)
        else:
            self.standalone.append(event)


@dataclass
class InstalledEtaShim:
    eta: float
    base_implementation_id: str
    implementation_id: str
    frozen_module: Any
    audit: EtaAudit
    original_screen_direction: Any
    original_select_direction: Any
    original_solve: Any
    runner_module: Any = None
    original_runner_implementation_id: Any = None
    original_runner_solve: Any = None
    runner_had_solve: bool = False
    original_atomic_save_npz: Any = None
    restored: bool = False

    def attach_runner(self, runner_module: Any) -> None:
        """Patch only runner metadata persistence, never runner numerics."""
        if self.runner_module is not None:
            raise RuntimeError("eta shim already has a runner")
        self.runner_module = runner_module
        self.original_runner_implementation_id = (
            runner_module.DISCRETE_B040_IMPLEMENTATION_ID
        )
        self.runner_had_solve = hasattr(
            runner_module, "solve_frozen_discrete_b040_band"
        )
        if self.runner_had_solve:
            self.original_runner_solve = (
                runner_module.solve_frozen_discrete_b040_band
            )
            runner_module.solve_frozen_discrete_b040_band = (
                self.frozen_module.solve_frozen_discrete_b040_band
            )
        self.original_atomic_save_npz = runner_module.atomic_save_npz
        runner_module.DISCRETE_B040_IMPLEMENTATION_ID = self.implementation_id

        def atomic_save_with_eta(destination: Path, **arrays: Any) -> None:
            frozen_subproblem = str(
                np.asarray(arrays.get("frozen_subproblem", "")).item()
            )
            if frozen_subproblem != "discrete_b040":
                raise RuntimeError(
                    "selected-step eta shim requires frozen_subproblem=discrete_b040"
                )
            result_id = str(
                np.asarray(arrays.get("discrete_implementation_id", "")).item()
            )
            if result_id != self.implementation_id:
                raise RuntimeError("eta-specific discrete implementation ID drift")
            selections = np.asarray(
                arrays.get("discrete_audit_selection", []), dtype="U8"
            )
            events = list(self.audit.committed)
            if len(events) != int(selections.size):
                raise RuntimeError(
                    "eta audit length differs from committed discrete selections"
                )
            event_labels = np.asarray([event.label for event in events], dtype="U8")
            if not np.array_equal(event_labels, selections):
                raise RuntimeError("eta audit labels differ from discrete selections")

            additions = {
                "discrete_step_multiplier_policy_id": np.array(SHIM_POLICY_ID),
                "discrete_step_multiplier_base_implementation_id": np.array(
                    self.base_implementation_id
                ),
                "discrete_step_multiplier_implementation_id": np.array(
                    self.implementation_id
                ),
                "discrete_step_multiplier_eta": np.array(self.eta),
                "discrete_step_multiplier_application_stage": np.array(
                    APPLICATION_STAGE
                ),
                "discrete_step_multiplier_selector_isolated": np.array(True),
                "discrete_step_multiplier_dynamic_cap_reapplied": np.array(True),
                "discrete_step_multiplier_energy_backtracking": np.array(False),
                "discrete_step_multiplier_extra_physical_evaluations": np.array(0),
                "discrete_step_multiplier_audit_selection": event_labels,
                "discrete_step_multiplier_audit_requested": np.asarray(
                    [event.requested_multiplier for event in events], dtype=float
                ),
                "discrete_step_multiplier_audit_applied": np.asarray(
                    [event.applied_multiplier for event in events], dtype=float
                ),
                "discrete_step_multiplier_audit_cap_hit": np.asarray(
                    [event.cap_hit for event in events], dtype=bool
                ),
                "discrete_step_multiplier_audit_dynamic_max_step": np.asarray(
                    [event.dynamic_max_step for event in events], dtype=float
                ),
                "discrete_step_multiplier_audit_base_max_atomic_displacement": (
                    np.asarray(
                        [
                            event.base_max_atomic_displacement
                            for event in events
                        ],
                        dtype=float,
                    )
                ),
                "discrete_step_multiplier_audit_requested_max_atomic_displacement": (
                    np.asarray(
                        [
                            event.requested_max_atomic_displacement
                            for event in events
                        ],
                        dtype=float,
                    )
                ),
                "discrete_step_multiplier_audit_applied_max_atomic_displacement": (
                    np.asarray(
                        [
                            event.applied_max_atomic_displacement
                            for event in events
                        ],
                        dtype=float,
                    )
                ),
                "discrete_step_multiplier_uncommitted_selection_count": np.array(
                    len(self.audit.uncommitted), dtype=int
                ),
                "discrete_step_multiplier_aborted_selection_count": np.array(
                    len(self.audit.aborted), dtype=int
                ),
            }
            collisions = sorted(set(additions).intersection(arrays))
            if collisions:
                raise RuntimeError(
                    "eta metadata key collision: " + ", ".join(collisions)
                )
            self.original_atomic_save_npz(
                destination, **{**arrays, **additions}
            )

        runner_module.atomic_save_npz = atomic_save_with_eta

    def restore(self) -> None:
        """Restore patched module attributes; primarily used by parity tests."""
        if self.restored:
            return
        if self.runner_module is not None:
            self.runner_module.DISCRETE_B040_IMPLEMENTATION_ID = (
                self.original_runner_implementation_id
            )
            if self.runner_had_solve:
                # A fresh runner imported after the frozen patch captures the
                # wrapper by value.  Restore its alias as well as the defining
                # frozen module so an in-process eta=1 parity test cannot leak.
                self.runner_module.solve_frozen_discrete_b040_band = (
                    self.original_solve
                )
            self.runner_module.atomic_save_npz = self.original_atomic_save_npz
        self.frozen_module.screen_direction = self.original_screen_direction
        self.frozen_module.select_direction = self.original_select_direction
        self.frozen_module.solve_frozen_discrete_b040_band = self.original_solve
        self.frozen_module.IMPLEMENTATION_ID = self.base_implementation_id
        self.restored = True


def install_frozen_eta_shim(frozen_module: Any, eta: float) -> InstalledEtaShim:
    """Install post-selection eta scaling into one frozen-module instance."""
    eta = validate_eta(eta)
    base_id = str(frozen_module.IMPLEMENTATION_ID)
    implementation_id = shim_implementation_id(base_id, eta)
    original_screen = frozen_module.screen_direction
    original_select = frozen_module.select_direction
    original_solve = frozen_module.solve_frozen_discrete_b040_band
    audit = EtaAudit()

    def screen_with_metadata(
        label: str,
        raw_direction: Any,
        gradient: Any,
        tangents: Any,
        max_step: float,
    ) -> Any:
        proposal = original_screen(
            label, raw_direction, gradient, tangents, max_step
        )
        # Metadata is private to this shim.  Numeric proposal fields remain
        # exactly those produced by frozen v1 and therefore determine the
        # original selector without eta contamination.
        proposal["_selected_eta_gradient"] = np.asarray(
            gradient, dtype=np.float64
        )
        proposal["_selected_eta_tangents"] = np.asarray(
            tangents, dtype=np.float64
        )
        proposal["_selected_eta_dynamic_max_step"] = float(max_step)
        return proposal

    def select_then_scale(h0_proposal: Any, lbfgs_proposal: Any) -> Any:
        selected = original_select(h0_proposal, lbfgs_proposal)
        try:
            gradient = selected["_selected_eta_gradient"]
            tangents = selected["_selected_eta_tangents"]
            dynamic_max_step = float(
                selected["_selected_eta_dynamic_max_step"]
            )
            base_step = np.asarray(selected["step"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "selected eta proposal is missing screen metadata"
            ) from error

        base_norm = float(np.linalg.norm(base_step))
        if not math.isfinite(base_norm) or base_norm <= 0.0:
            raise RuntimeError("selected base step is non-finite or zero")
        base_max = _max_atomic_displacement(base_step)
        requested_step = eta * base_step
        requested_max = _max_atomic_displacement(requested_step)

        if eta == 1.0:
            # Exact parity contract: do not copy, project, cap, or rewrite any
            # original proposal field when eta is binary64 one.
            applied_step = base_step
            applied_multiplier = 1.0
            cap_hit = False
            returned = selected
        else:
            applied_step, _ = frozen_module.cap_joint_direction(
                requested_step, tangents, dynamic_max_step
            )
            applied_multiplier = float(np.linalg.norm(applied_step) / base_norm)
            if (
                not math.isfinite(applied_multiplier)
                or applied_multiplier <= 0.0
                or applied_multiplier > eta * (1.0 + 1.0e-12)
            ):
                raise RuntimeError("invalid applied selected-step multiplier")
            cap_hit = bool(requested_max > dynamic_max_step)
            capped_decrease = -float(np.sum(gradient * applied_step))
            if not math.isfinite(capped_decrease) or capped_decrease <= 0.0:
                raise RuntimeError("eta-adjusted selected step is not descent")
            returned = dict(selected)
            returned["step"] = applied_step
            returned["capped_decrease"] = capped_decrease

        event = EtaEvent(
            label=str(selected["label"]),
            requested_multiplier=eta,
            applied_multiplier=applied_multiplier,
            cap_hit=cap_hit,
            dynamic_max_step=dynamic_max_step,
            base_max_atomic_displacement=base_max,
            requested_max_atomic_displacement=requested_max,
            applied_max_atomic_displacement=_max_atomic_displacement(
                applied_step
            ),
        )
        audit.record(event)
        return returned

    def solve_with_committed_audit(*args: Any, **kwargs: Any) -> Any:
        local_events: List[EtaEvent] = []
        audit.capture_stack.append(local_events)
        try:
            result = original_solve(*args, **kwargs)
        except BaseException:
            audit.aborted.extend(local_events)
            raise
        finally:
            popped = audit.capture_stack.pop()
            if popped is not local_events:
                raise RuntimeError("eta audit capture stack corruption")
        report = result[8]
        committed_count = int(report["joint_steps"])
        if committed_count < 0 or committed_count > len(local_events):
            raise RuntimeError("eta audit cannot identify committed selections")
        audit.committed.extend(local_events[:committed_count])
        audit.uncommitted.extend(local_events[committed_count:])
        return result

    frozen_module.screen_direction = screen_with_metadata
    frozen_module.select_direction = select_then_scale
    frozen_module.solve_frozen_discrete_b040_band = solve_with_committed_audit
    frozen_module.IMPLEMENTATION_ID = implementation_id
    return InstalledEtaShim(
        eta=eta,
        base_implementation_id=base_id,
        implementation_id=implementation_id,
        frozen_module=frozen_module,
        audit=audit,
        original_screen_direction=original_screen,
        original_select_direction=original_select,
        original_solve=original_solve,
    )


def _assert_exact_nested(reference: Any, candidate: Any, path: str = "root") -> None:
    if isinstance(reference, np.ndarray) or isinstance(candidate, np.ndarray):
        if not np.array_equal(np.asarray(reference), np.asarray(candidate)):
            raise AssertionError(f"eta=1 parity differs at {path}")
        return
    if isinstance(reference, dict) and isinstance(candidate, dict):
        if reference.keys() != candidate.keys():
            raise AssertionError(f"eta=1 mapping keys differ at {path}")
        for key in reference:
            if key == "implementation_id":
                continue
            _assert_exact_nested(
                reference[key], candidate[key], f"{path}.{key}"
            )
        return
    if isinstance(reference, (list, tuple)) and isinstance(
        candidate, (list, tuple)
    ):
        if type(reference) is not type(candidate) or len(reference) != len(candidate):
            raise AssertionError(f"eta=1 sequence differs at {path}")
        for index, (left, right) in enumerate(zip(reference, candidate)):
            _assert_exact_nested(left, right, f"{path}[{index}]")
        return
    if type(reference) is not type(candidate) or reference != candidate:
        raise AssertionError(f"eta=1 parity differs at {path}")


def assert_eta_one_exact_parity(reference: Any, candidate: Any) -> None:
    """Assert solver-output identity except for the registered implementation ID."""
    _assert_exact_nested(reference, candidate)


def main(argv: Optional[Sequence[str]] = None) -> None:
    process_argv = list(sys.argv if argv is None else argv)
    eta, runner_argv = extract_eta_argument(process_argv)
    sys.argv = runner_argv

    frozen = importlib.import_module("frozen_discrete_b040")
    installed = install_frozen_eta_shim(frozen, eta)
    # Import after the frozen ID/function patch so the runner's from-imported
    # aliases see the eta-specific implementation on first load.
    runner = importlib.import_module(
        "run_ivm_v9_spring_projected_production_20260730"
    )
    installed.attach_runner(runner)
    runner.main()


if __name__ == "__main__":
    main()
