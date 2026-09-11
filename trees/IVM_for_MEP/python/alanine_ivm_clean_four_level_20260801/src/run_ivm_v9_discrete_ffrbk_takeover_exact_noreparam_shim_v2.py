#!/usr/bin/env python3
"""Signature-safe v2 adapter for the exact/no-reparameterization runtime.

The registered v1 runtime composes the candidate-aware solver, selector, and
selected-step eta wrappers before installing its solve-completion audit.  The
eta wrapper intentionally exposes ``*args, **kwargs``.  Inspecting that outer
wrapper therefore loses the candidate solver's named call contract, even
though the unchanged production runner still calls the solver with the
candidate solver's eight positional arguments.

This adapter freezes the explicit base/candidate-compatible signature before
composition and supplies it only while the v1 completion audit captures its
argument binder.  It does not alter solver numerics or any registered v1
source.  A distinct runtime policy and implementation token make v2 artifacts
unambiguous continuations rather than silent replacements of v1 artifacts.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Optional, Sequence

import run_ivm_v9_discrete_ffrbk_takeover_exact_noreparam_shim as legacy


POLICY_ID = (
    "ala-discrete-exactlike-r01-no-reparam-alpha-grid-runtime-"
    "v2-call-signature"
)
IMPLEMENTATION_POLICY_ID = "ala-exact-completion-audit-call-signature-v2"
SUPERSEDED_RUNTIME_POLICY_ID = legacy.POLICY_ID

# Re-export the scientific contract used by a successor member/auditor.  These
# are aliases, not relaxed or independently reimplemented controls.
FAMILY = legacy.FAMILY
OPERATIONAL_DEFINITION = legacy.OPERATIONAL_DEFINITION
EXACT_COMPLETE_MASK_KEY = legacy.EXACT_COMPLETE_MASK_KEY
COMPLETE_BAND_PREFLIGHT_POLICY_ID = legacy.COMPLETE_BAND_PREFLIGHT_POLICY_ID
RESIDUAL_RATIO = legacy.RESIDUAL_RATIO
INNER_MAX_STEPS = legacy.INNER_MAX_STEPS
TOTAL_INTERNAL_NF_BUDGET = legacy.TOTAL_INTERNAL_NF_BUDGET
PER_IMAGE_SAFETY_BUDGET = legacy.PER_IMAGE_SAFETY_BUDGET
PHYSICAL_MAX_STEP = legacy.PHYSICAL_MAX_STEP
ETA = legacy.ETA
HISTORY = legacy.HISTORY
KKT_ATOL = legacy.KKT_ATOL
STOP_MODE = legacy.STOP_MODE

validate_exact_no_reparam_runner_argv = (
    legacy.validate_exact_no_reparam_runner_argv
)
restore_runner_candidate_identity = legacy.restore_runner_candidate_identity


class ExactNoReparamV2SafetyError(RuntimeError):
    """Fail closed if the solver call contract or adapter state drifts."""


POSITIONAL_ARGUMENTS = (
    "path",
    "energies",
    "forces",
    "tangents",
    "evaluators",
    "alpha",
    "max_step",
    "max_new_calls",
)
KEYWORD_ONLY_ARGUMENTS = (
    "inner_max_steps",
    "stop_mode",
    "relative_residual",
    "history_size",
    "kkt_atol",
    "preconditioner_bank",
    "active_images",
)


def validate_candidate_call_signature(
    signature: inspect.Signature,
) -> inspect.Signature:
    """Require the exact candidate/base solver's full named call contract."""

    if not isinstance(signature, inspect.Signature):
        raise TypeError("candidate call contract must be inspect.Signature")
    parameters = tuple(signature.parameters.values())
    expected_names = POSITIONAL_ARGUMENTS + KEYWORD_ONLY_ARGUMENTS
    if tuple(parameter.name for parameter in parameters) != expected_names:
        raise ExactNoReparamV2SafetyError(
            "candidate solver argument names/order drifted"
        )
    positional_kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    keyword_kind = inspect.Parameter.KEYWORD_ONLY
    if any(parameter.kind is not positional_kind for parameter in parameters[:8]):
        raise ExactNoReparamV2SafetyError(
            "candidate solver's first eight arguments must be positional/keyword"
        )
    if any(parameter.kind is not keyword_kind for parameter in parameters[8:]):
        raise ExactNoReparamV2SafetyError(
            "candidate solver controls must remain keyword-only"
        )
    if any(
        parameter.default is not inspect.Parameter.empty
        for parameter in parameters[:8]
    ):
        raise ExactNoReparamV2SafetyError(
            "candidate solver positional arguments must remain required"
        )
    return signature


def candidate_call_signature(callable_or_signature: Any) -> inspect.Signature:
    """Return and validate a callable's explicit candidate call signature."""

    signature = (
        callable_or_signature
        if isinstance(callable_or_signature, inspect.Signature)
        else inspect.signature(callable_or_signature)
    )
    return validate_candidate_call_signature(signature)


_LEGACY_COMPLETION_INSTALLER = legacy.install_exact_solve_completion_audit


def install_exact_solve_completion_audit(
    frozen_module: Any,
    argument_contract: Any,
) -> legacy.InstalledExactSolveCompletionAudit:
    """Install the unchanged v1 audit with an explicit solver signature.

    ``argument_contract`` is captured before selector/eta composition.  The
    outer solve function receives a temporary ``__signature__`` only for the
    duration of the legacy install call; its previous attribute is restored
    before a physical solver call can occur.
    """

    signature = candidate_call_signature(argument_contract)
    outer_solve = frozen_module.solve_frozen_discrete_b040_band
    missing = object()
    previous = getattr(outer_solve, "__signature__", missing)
    try:
        setattr(outer_solve, "__signature__", signature)
    except (AttributeError, TypeError) as error:
        raise ExactNoReparamV2SafetyError(
            "outer frozen solver does not admit a temporary call signature"
        ) from error
    try:
        installed = _LEGACY_COMPLETION_INSTALLER(frozen_module)
    finally:
        if previous is missing:
            try:
                delattr(outer_solve, "__signature__")
            except AttributeError as error:
                raise ExactNoReparamV2SafetyError(
                    "temporary outer-solver signature was lost"
                ) from error
        else:
            setattr(outer_solve, "__signature__", previous)
    if installed.original_solve is not outer_solve:
        raise ExactNoReparamV2SafetyError(
            "completion audit wrapped a different frozen solver"
        )
    return installed


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the v1 scientific composition through the signature-safe adapter."""

    if legacy.POLICY_ID != SUPERSEDED_RUNTIME_POLICY_ID:
        raise ExactNoReparamV2SafetyError("legacy runtime policy drifted")
    if (
        legacy.install_exact_solve_completion_audit
        is not _LEGACY_COMPLETION_INSTALLER
    ):
        raise ExactNoReparamV2SafetyError(
            "legacy completion installer is already patched"
        )

    frozen = importlib.import_module("frozen_discrete_b040")
    original_implementation_id = str(frozen.IMPLEMENTATION_ID)
    if not original_implementation_id:
        raise ExactNoReparamV2SafetyError(
            "base discrete implementation ID is empty"
        )
    signature = candidate_call_signature(
        frozen.solve_frozen_discrete_b040_band
    )
    composed_base_id = "+".join(
        (original_implementation_id, IMPLEMENTATION_POLICY_ID)
    )

    original_policy_id = legacy.POLICY_ID
    original_installer = legacy.install_exact_solve_completion_audit

    def install_with_frozen_contract(
        patched_frozen_module: Any,
    ) -> legacy.InstalledExactSolveCompletionAudit:
        if patched_frozen_module is not frozen:
            raise ExactNoReparamV2SafetyError(
                "completion audit received an unexpected frozen module"
            )
        return install_exact_solve_completion_audit(
            patched_frozen_module, signature
        )

    legacy.POLICY_ID = POLICY_ID
    legacy.install_exact_solve_completion_audit = install_with_frozen_contract
    frozen.IMPLEMENTATION_ID = composed_base_id
    try:
        legacy.main(argv)
    finally:
        frozen.IMPLEMENTATION_ID = original_implementation_id
        legacy.install_exact_solve_completion_audit = original_installer
        legacy.POLICY_ID = original_policy_id


if __name__ == "__main__":
    main()
