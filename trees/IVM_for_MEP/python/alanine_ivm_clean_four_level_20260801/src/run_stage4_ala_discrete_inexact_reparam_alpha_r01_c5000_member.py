#!/usr/bin/env python3
"""Run one registered Ala alpha-sensitivity member at r=0.1, c=5000.

The scientific command is cloned token-for-token from the audited Stage-IV
one-way run.  It changes only alpha, the registered fixed residual ratio and
implementation safety cap, the shared total-internal-force budget, the
runtime adapter, and exclusive output routing.  Reparameterization remains
``state_curve/per_image`` after every outer proposal.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Tuple

import numpy as np

import run_ivm_v9_discrete_ffrbk_takeover_alpha_r01_c5000_shim as runtime
import run_ivm_v9_discrete_ffrbk_takeover_grid_shim as grid_runtime
import run_ivm_v9_total_internal_nf_budget as total_budget
import run_stage4_ala_discrete_inexact_reparam_grid_member as frozen_member
import run_stage4_uniform_optimized as legacy


MEMBER_POLICY_VERSION = "stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_v1"
CANONICAL_REPLICATE = "r1"
TOTAL_INTERNAL_NF_BUDGET = 5000
RESIDUAL_RATIO = 0.1
INNER_MAX_STEPS = 5000
INNER_CAP_STOP_TOKEN = "evaluated-inner-max-steps"
ALPHAS = runtime.ALPHAS
CANONICAL_COMMAND = frozen_member.CANONICAL_COMMAND
CANONICAL_COMMAND_SHA256 = frozen_member.CANONICAL_COMMAND_SHA256
ROUTING_FLAGS = frozenset(("--output-dir", "--run-label"))
SCIENTIFIC_VALUE_FLAGS = frozenset(
    (
        "--alpha",
        "--budget",
        "--discrete-inner-max-steps",
        "--discrete-residual-ratio",
    )
)


class MemberSafetyError(RuntimeError):
    """Fail-closed error for command, result, or cap-binding drift."""


def file_sha256(path: Path) -> str:
    return frozen_member.file_sha256(path)


def compact_command_sha256(command: List[str]) -> str:
    return frozen_member.compact_command_sha256(command)


def registered_alpha(alpha: Any) -> float:
    value = float(alpha)
    if not math.isfinite(value) or value not in ALPHAS:
        raise ValueError("alpha is not in 1e-1,...,1e6")
    return value


def alpha_tag(alpha: Any) -> str:
    value = registered_alpha(alpha)
    exponent = int(round(math.log10(value)))
    return "a1e" + ("m" + str(-exponent) if exponent < 0 else str(exponent))


def run_label(alpha: Any) -> str:
    return "_".join(
        (
            MEMBER_POLICY_VERSION,
            alpha_tag(alpha),
            "rr10",
            "c5000",
            "tnf5000",
            CANONICAL_REPLICATE,
        )
    )


def controls(alpha: Any) -> grid_runtime.GridControls:
    return runtime.validate_controls(
        grid_runtime.GridControls(
            alpha=registered_alpha(alpha),
            residual_ratio=RESIDUAL_RATIO,
            inner_max_steps=INNER_MAX_STEPS,
            total_internal_nf_budget=TOTAL_INTERNAL_NF_BUDGET,
        )
    )


def _options(command: List[str]) -> Dict[str, str]:
    return frozen_member._options(command)


def assert_registered_command(
    command: List[str], baseline: List[str], cell: grid_runtime.GridControls
) -> None:
    cell = runtime.validate_controls(cell)
    if len(command) != len(baseline) + 2:
        raise MemberSafetyError("member command length differs from frozen clone")
    if command[-2:] != [
        total_budget.TOTAL_INTERNAL_NF_FLAG,
        str(TOTAL_INTERNAL_NF_BUDGET),
    ]:
        raise MemberSafetyError("total-internal-NF option is missing or misplaced")
    allowed_indices = {1}
    for flag in ROUTING_FLAGS | SCIENTIFIC_VALUE_FLAGS:
        occurrences = [index for index, token in enumerate(baseline) if token == flag]
        if len(occurrences) != 1:
            raise MemberSafetyError(f"frozen command has invalid {flag}")
        allowed_indices.add(occurrences[0] + 1)
    drift = [
        index
        for index, (observed, frozen) in enumerate(zip(command[:-2], baseline))
        if observed != frozen and index not in allowed_indices
    ]
    if drift:
        raise MemberSafetyError(
            "member command changed frozen token(s): "
            + ",".join(str(index) for index in drift)
        )
    root = Path(_options(command)["--root"]).expanduser().resolve()
    expected_runtime = str(
        (root / "src/run_ivm_v9_discrete_ffrbk_takeover_alpha_r01_c5000_shim.py").resolve()
    )
    if command[1] != expected_runtime:
        raise MemberSafetyError("member command uses the wrong runtime adapter")
    options = _options(command)
    exact = {
        "--solver": "state_ivm",
        "--alpha": f"{cell.alpha:g}",
        "--alpha-mode": "fixed",
        "--alpha-max": "inf",
        "--max-step": "1",
        "--history": "7",
        "--budget": "5000",
        "--state-globalization": "iperp_monotone",
        "--reparam-mode": "state_curve",
        "--reparam-every": "1",
        "--state-curve-direct-objective-acceptance": "per_image",
        "--stop-complete-iperp": "-1",
        "--frozen-subproblem": "discrete_b040",
        "--discrete-inner-max-steps": "5000",
        "--discrete-stop-mode": "r08c10",
        "--discrete-residual-ratio": "0.1",
        "--discrete-history": "7",
        "--discrete-step-multiplier-eta": "1.1",
        "--plbfgs-kkt-atol": "0",
        "--discrete-ffrbk-selector-mode": "one_way_takeover",
        total_budget.TOTAL_INTERNAL_NF_FLAG: "5000",
    }
    for flag, expected in exact.items():
        if options.get(flag) != expected:
            raise MemberSafetyError(f"member command drifted at {flag}")
    if options.get("--state-curve-direct-solve") != "":
        raise MemberSafetyError("member command disabled state-curve direct solve")


def single_run_command(root: Path, alpha: Any) -> Tuple[List[str], str]:
    root = root.expanduser().resolve()
    cell = controls(alpha)
    baseline = frozen_member.canonical_command(root)
    command = list(baseline)
    command[1] = str(
        (root / "src/run_ivm_v9_discrete_ffrbk_takeover_alpha_r01_c5000_shim.py").resolve()
    )
    label = run_label(cell.alpha)
    frozen_member._replace_option(
        command, "--output-dir", f"runs/{MEMBER_POLICY_VERSION}/members"
    )
    frozen_member._replace_option(command, "--run-label", label)
    frozen_member._replace_option(command, "--alpha", f"{cell.alpha:g}")
    frozen_member._replace_option(command, "--budget", "5000")
    frozen_member._replace_option(
        command, "--discrete-inner-max-steps", "5000"
    )
    frozen_member._replace_option(
        command, "--discrete-residual-ratio", "0.1"
    )
    command.extend([total_budget.TOTAL_INTERNAL_NF_FLAG, "5000"])
    assert_registered_command(command, baseline, cell)
    return command, label


def member_paths(root: Path, alpha: Any) -> Dict[str, Any]:
    root = root.expanduser().resolve()
    label = run_label(alpha)
    return {
        "label": label,
        "result_dir": root / "runs" / MEMBER_POLICY_VERSION / "members",
        "result_glob": f"ivm_{label}_central_state_ivm_cfg*_budget5000.npz",
        "companion_glob": (
            f"ivm_{label}_central_state_ivm_cfg*_budget5000.audit.json"
        ),
        "audit": root / "runs" / MEMBER_POLICY_VERSION / "audits" / f"{label}.json",
        "member_log": root / "logs" / MEMBER_POLICY_VERSION / "members" / label,
    }


@contextmanager
def _frozen_validation_context() -> Iterator[None]:
    previous_caps = frozen_member.INNER_MAX_STEPS
    previous_policy_id = grid_runtime.POLICY_ID
    frozen_member.INNER_MAX_STEPS = (INNER_MAX_STEPS,)
    grid_runtime.POLICY_ID = runtime.POLICY_ID
    try:
        yield
    finally:
        grid_runtime.POLICY_ID = previous_policy_id
        frozen_member.INNER_MAX_STEPS = previous_caps


def validate_result(
    result: Path, cell: grid_runtime.GridControls
) -> Dict[str, Any]:
    cell = runtime.validate_controls(cell)
    with _frozen_validation_context():
        summary = frozen_member.validate_result(result, cell)
    with np.load(result, allow_pickle=False) as saved:
        reasons = np.asarray(saved["discrete_audit_stop_reason"], dtype="U64")
        inner_cap_stop_count = int(np.count_nonzero(reasons == INNER_CAP_STOP_TOKEN))
        if inner_cap_stop_count != 0:
            raise MemberSafetyError(
                "implementation safety cap bound the fixed solve: "
                f"{inner_cap_stop_count} stop(s)"
            )
        if int(np.asarray(saved["discrete_inner_max_steps"]).item()) != 5000:
            raise MemberSafetyError("saved inner max steps is not 5000")
    return {
        **summary,
        "inner_cap_stop_token": INNER_CAP_STOP_TOKEN,
        "inner_cap_stop_count": 0,
        "inner_cap_empirically_nonbinding": True,
    }


def validate_companion(
    result: Path, final_internal_nf: int, blocked: int
) -> Dict[str, Any]:
    return frozen_member.validate_companion(result, final_internal_nf, blocked)


def validate_existing(root: Path, alpha: Any) -> Dict[str, Any]:
    root = root.expanduser().resolve()
    cell = controls(alpha)
    command, _ = single_run_command(root, cell.alpha)
    paths = member_paths(root, cell.alpha)
    results = sorted(paths["result_dir"].glob(paths["result_glob"]))
    companions = sorted(paths["result_dir"].glob(paths["companion_glob"]))
    audit_path = paths["audit"]
    log_dir = paths["member_log"]
    if (
        len(results) != 1
        or len(companions) != 1
        or not audit_path.is_file()
        or not log_dir.is_dir()
    ):
        raise MemberSafetyError(f"member is absent or partial: {paths['label']}")
    result = results[0].resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "complete":
        raise MemberSafetyError("member audit status is not complete")
    if audit.get("command") != command:
        raise MemberSafetyError("member audit command differs from registration")
    if Path(str(audit.get("result", ""))).resolve() != result:
        raise MemberSafetyError("member audit points to another result")
    if str(audit.get("result_sha256")) != file_sha256(result):
        raise MemberSafetyError("member result hash differs from its audit")
    summary = validate_result(result, cell)
    companion = validate_companion(
        result,
        int(summary["final_optimization_NF"]),
        int(summary["total_budget_blocked_miss_count"]),
    )
    if str(audit.get("companion_audit_sha256")) != str(
        companion["companion_audit_sha256"]
    ):
        raise MemberSafetyError("companion hash differs from member audit")
    return {
        "alpha": cell.alpha,
        "result": str(result),
        "result_sha256": file_sha256(result),
        "member_audit": str(audit_path.resolve()),
        "member_audit_sha256": file_sha256(audit_path),
        **summary,
        **companion,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alpha", required=True)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    root = args.root.expanduser().resolve()
    cell = controls(args.alpha)
    command, label = single_run_command(root, cell.alpha)
    paths = member_paths(root, cell.alpha)
    result_dir = paths["result_dir"]
    audit_path = paths["audit"]
    log_dir = paths["member_log"]
    existing = sorted(result_dir.glob(paths["result_glob"])) + sorted(
        result_dir.glob(paths["companion_glob"])
    )
    if existing or audit_path.exists() or log_dir.exists():
        target = existing[0] if existing else audit_path if audit_path.exists() else log_dir
        raise FileExistsError(f"refusing to overwrite alpha member: {target}")
    result_dir.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    wall_seconds = legacy.run_command(
        command, root, log_dir / "stdout.txt", log_dir / "stderr.txt"
    )
    result = legacy.find_result(result_dir, label, TOTAL_INTERNAL_NF_BUDGET)
    summary = validate_result(result, cell)
    companion = validate_companion(
        result,
        int(summary["final_optimization_NF"]),
        int(summary["total_budget_blocked_miss_count"]),
    )
    audit: Mapping[str, Any] = {
        "status": "complete",
        "member_policy_version": MEMBER_POLICY_VERSION,
        "runtime_policy_id": runtime.POLICY_ID,
        "alpha": cell.alpha,
        "residual_ratio": RESIDUAL_RATIO,
        "inner_max_steps": INNER_MAX_STEPS,
        "inner_cap_acceptance_gate": "inner_cap_stop_count == 0",
        "stop_rule": "R/R0 <= 0.1; c5000 is an implementation safety cap",
        "physical_max_step_angstrom": 1.0,
        "eta": 1.1,
        "kkt_atol": 0.0,
        "lbfgs_history": 7,
        "total_internal_nf_budget": TOTAL_INTERNAL_NF_BUDGET,
        "endpoint_evaluations_excluded_from_optimization_nf": True,
        "reparameterization_policy": "state_curve/per_image from first state",
        "complete_state_definition": "initial_or_reparam_committed",
        "scientific_command_sha256": compact_command_sha256(command),
        "audited_3389_command_sha256": CANONICAL_COMMAND_SHA256,
        "result": str(result.resolve()),
        "result_sha256": file_sha256(result),
        "wall_seconds": wall_seconds,
        "scientific_run_converged": False,
        "command": command,
        **summary,
        **companion,
    }
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
