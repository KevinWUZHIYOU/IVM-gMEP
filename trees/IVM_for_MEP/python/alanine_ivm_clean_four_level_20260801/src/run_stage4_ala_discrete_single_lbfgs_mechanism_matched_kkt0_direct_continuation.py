#!/usr/bin/env python3
"""Run the preregistered direct-LBFGS-only continuation cell.

The scientific command is the original v1 registered ``direct_lbfgs_only``
command byte-for-byte except for ``--output-dir`` and ``--run-label``.  This
wrapper accepts a valid trajectory with 1--254 records.  A trajectory that
ends before record 254 is retained as nonconverged and all precision beyond
its tightest reached value is treated as unreachable.

Only a fixed I_perp/mechanism allowlist is loaded.  Energy and barrier fields
are never indexed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import run_stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0 as base
import run_stage4_uniform_optimized as legacy


MEMBER_POLICY_VERSION = (
    "stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_"
    "direct_continuation_v1"
)
ARM = "direct_lbfgs_only"
ORIGINAL_PREREGISTRATION = (
    "reports/stage4_ala_discrete_single_lbfgs_mechanism_matched_kkt0_v1_"
    "N254_r1_1787260497802476000_preregistered.json"
)
ORIGINAL_PREREGISTRATION_SHA256 = (
    "04de96743da079b8b35247214a4ad3cf387157edb2134381cc08f95d3bab1f86"
)
ORIGINAL_DIRECT_COMMAND_SHA256 = (
    "8b41d35618a3c3d93f50fb56122e7b0751fc4f3b297698fb24b1a008225ca167"
)
ROUTING_OPTIONS = ("--output-dir", "--run-label")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact_command_sha256(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_label() -> str:
    return (
        f"{MEMBER_POLICY_VERSION}_{ARM}_eta1p1_r08c10_kkt0_a15_"
        f"N254_discrete_b040_{base.CANONICAL_REPLICATE}"
    )


def member_paths(root: Path) -> dict[str, Path | str]:
    label = run_label()
    return {
        "label": label,
        "result_dir": root / "runs" / MEMBER_POLICY_VERSION / "members",
        "audit": root / "runs" / MEMBER_POLICY_VERSION / "audits" / f"{label}.json",
        "member_log": root / "logs" / MEMBER_POLICY_VERSION / "members" / label,
        "launcher_log": root / "logs" / MEMBER_POLICY_VERSION / "launcher" / label,
    }


def _original_direct_command(root: Path) -> list[str]:
    preregistration = (root / ORIGINAL_PREREGISTRATION).resolve()
    if _file_sha256(preregistration) != ORIGINAL_PREREGISTRATION_SHA256:
        raise RuntimeError("original v1 preregistration hash drifted")
    payload = json.loads(preregistration.read_text(encoding="utf-8"))
    record = payload.get("registered_commands", {}).get(ARM)
    if not isinstance(record, dict):
        raise RuntimeError("original direct command record is missing")
    command = record.get("command")
    if not isinstance(command, list) or not all(
        isinstance(token, str) for token in command
    ):
        raise RuntimeError("original direct command is not a string vector")
    command_hash = _compact_command_sha256(command)
    if (
        record.get("compact_sha256") != ORIGINAL_DIRECT_COMMAND_SHA256
        or command_hash != ORIGINAL_DIRECT_COMMAND_SHA256
    ):
        raise RuntimeError("original direct command hash drifted")
    return command


def _replace_option(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"expected exactly one valued {flag}")
    command[positions[0] + 1] = value


def assert_registered_command(
    command: list[str], original: list[str], root: Path
) -> None:
    expected = list(original)
    _replace_option(
        expected,
        "--output-dir",
        f"runs/{MEMBER_POLICY_VERSION}/members",
    )
    _replace_option(expected, "--run-label", run_label())
    if command != expected:
        raise RuntimeError("continuation changed more than registered routing")

    if len(original) != len(command):
        raise RuntimeError("continuation command length drifted")
    differing_indices = [
        index
        for index, (old, new) in enumerate(zip(original, command))
        if old != new
    ]
    allowed_indices = sorted(
        original.index(flag) + 1 for flag in ROUTING_OPTIONS
    )
    if differing_indices != allowed_indices:
        raise RuntimeError("continuation command difference audit failed")

    options = base.command_options(command)
    required = {
        "--root": str(root.resolve()),
        "--initial-path-file": "inputs/common_initial_path_equidistant.npz",
        "--solver": "state_ivm",
        "--alpha": "15",
        "--alpha-mode": "fixed",
        "--budget": "254",
        "--reparam-mode": "state_curve",
        "--state-curve-direct-objective-acceptance": "per_image",
        "--frozen-subproblem": "discrete_b040",
        "--discrete-inner-max-steps": "10",
        "--discrete-stop-mode": "r08c10",
        "--discrete-residual-ratio": "0.8",
        "--discrete-history": "7",
        "--discrete-step-multiplier-eta": "1.1",
        "--plbfgs-kkt-atol": "0",
        "--discrete-scalar-lbfgs-mode": ARM,
    }
    for flag, value in required.items():
        if options.get(flag) != value:
            raise RuntimeError(f"continuation command drifted at {flag}")


def single_run_command(root: Path) -> tuple[list[str], str]:
    original = _original_direct_command(root)
    command = list(original)
    _replace_option(
        command,
        "--output-dir",
        f"runs/{MEMBER_POLICY_VERSION}/members",
    )
    _replace_option(command, "--run-label", run_label())
    assert_registered_command(command, original, root)
    return command, run_label()


def _scalar(saved: np.lib.npyio.NpzFile, key: str):
    return np.asarray(saved[key]).item()


def _validate_common_flexible(saved: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if float(_scalar(saved, "alpha")) != base.CANONICAL_ALPHA or str(
        _scalar(saved, "alpha_mode")
    ) != "fixed":
        raise RuntimeError("direct continuation changed fixed alpha")
    active = np.asarray(saved["active_alpha"], dtype=float)
    record_count = int(active.size)
    if (
        active.ndim != 1
        or not 1 <= record_count <= base.CANONICAL_BUDGET
        or not np.all(active == base.CANONICAL_ALPHA)
    ):
        raise RuntimeError("direct continuation trajectory must have 1--254 alpha15 records")

    expected = {
        "frozen_subproblem": "discrete_b040",
        "discrete_inner_max_steps": base.CANONICAL_MAX_EVALUATED_STEPS,
        "discrete_stop_mode": "r08c10",
        "discrete_residual_ratio": base.CANONICAL_RATIO,
        "discrete_history": base.CANONICAL_HISTORY,
        "plbfgs_kkt_atol": base.CANONICAL_KKT_ATOL,
        "reparam_mode": "state_curve",
        "state_curve_direct_objective_acceptance": "per_image",
        "stop_complete_iperp": -1.0,
        "initial_path_sha256": base.COMMON_INITIAL_ARRAY_SHA256,
    }
    for key, value in expected.items():
        if _scalar(saved, key) != value:
            raise RuntimeError(f"direct continuation result drifted at {key}")
    if not bool(_scalar(saved, "discrete_all_returned_states_evaluated")):
        raise RuntimeError("direct continuation returned an unevaluated state")

    backtracks = np.asarray(saved["subproblem_line_search_backtracks"], dtype=int)
    if backtracks.shape != (record_count,) or np.any(backtracks != 0):
        raise RuntimeError("direct continuation contains energy backtracking")

    cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
    i_perp = np.asarray(saved["i_perp"], dtype=float)
    committed = np.asarray(saved["reparam_committed"], dtype=bool)
    cv = np.asarray(saved["segment_length_cv"], dtype=float)
    ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
    arrays = (cost, i_perp, committed, cv, ratio)
    if any(array.shape != (record_count,) for array in arrays):
        raise RuntimeError("direct continuation trajectory arrays are misaligned")
    if np.any(np.diff(cost) <= 0) or not all(
        np.all(np.isfinite(array)) for array in (i_perp, cv, ratio)
    ):
        raise RuntimeError("direct continuation trajectory axes are invalid")

    selection = np.asarray(saved["discrete_audit_selection"], dtype="U8")
    outer = np.asarray(saved["discrete_audit_outer"], dtype=int)
    step = np.asarray(saved["discrete_audit_step"], dtype=int)
    if selection.shape != outer.shape or step.shape != outer.shape:
        raise RuntimeError("direct continuation discrete step audit is misaligned")
    ordered_outers = list(dict.fromkeys(outer.tolist()))
    reasons = np.asarray(saved["discrete_audit_stop_reason"], dtype="U64")
    relatives = np.asarray(
        saved["discrete_audit_global_relative_residual"], dtype=float
    )
    if reasons.shape != relatives.shape or reasons.size != len(ordered_outers):
        raise RuntimeError("direct continuation stop audit is misaligned")
    allowed_reasons = {
        "evaluated-relative-natural-residual",
        "evaluated-inner-max-steps",
        "per-image-new-call-limit",
        "preflight-per-image-new-call-limit",
    }
    if any(reason not in allowed_reasons for reason in reasons.tolist()):
        raise RuntimeError("direct continuation used a nonregistered inner stop")
    for index, outer_value in enumerate(ordered_outers):
        local_steps = step[outer == outer_value]
        if not np.array_equal(local_steps, np.arange(local_steps.size)):
            raise RuntimeError("direct continuation inner step numbering drifted")
        if local_steps.size > base.CANONICAL_MAX_EVALUATED_STEPS:
            raise RuntimeError("direct continuation exceeded k=10")
        reason = str(reasons[index])
        if reason == "evaluated-relative-natural-residual" and not (
            relatives[index] <= base.CANONICAL_RATIO
        ):
            raise RuntimeError("direct continuation relative stop occurred above r=0.8")
        if reason == "evaluated-inner-max-steps" and local_steps.size != 10:
            raise RuntimeError("direct continuation kmax stop did not occur at k=10")
    if any("absolute" in reason for reason in reasons.astype(str).tolist()):
        raise RuntimeError("direct continuation used absolute-KKT stopping")

    physical_stop_reason = str(_scalar(saved, "stop_reason"))
    if not physical_stop_reason:
        raise RuntimeError("direct continuation has an empty physical stop reason")
    early_stop = record_count < base.CANONICAL_BUDGET
    complete = committed.copy()
    complete[0] = True
    eligible = complete & (cv <= base.MESH_CV_MAX) & (ratio <= base.MESH_RATIO_MAX)
    budget_truncations = int(
        np.count_nonzero(
            np.isin(
                reasons,
                ["per-image-new-call-limit", "preflight-per-image-new-call-limit"],
            )
        )
    )
    return {
        "configuration_sha256": str(_scalar(saved, "configuration_sha256")),
        "initial_path_sha256": str(_scalar(saved, "initial_path_sha256")),
        "trajectory_record_count": record_count,
        "trajectory_record_count_allowed_range": [1, base.CANONICAL_BUDGET],
        "trajectory_complete_to_budget": not early_stop,
        "early_stop": early_stop,
        "physical_stop_reason": physical_stop_reason,
        "termination_class": (
            "early_stop_nonconverged"
            if early_stop
            else "budget_complete_not_convergence"
        ),
        "scientific_run_converged": False,
        "early_stop_counted_as_converged": False,
        "precision_beyond_tightest_reached_unreachable": early_stop,
        "unreachable_threshold_policy": "candidate_unreachable_is_loss",
        "final_optimization_NF": int(cost[-1]),
        "complete_state_NF": cost[complete].tolist(),
        "mesh_eligible_complete_state_NF": cost[eligible].tolist(),
        "mesh_eligible_complete_state_count": int(np.count_nonzero(eligible)),
        "inner_subproblem_count": int(reasons.size),
        "relative_stop_count": int(
            np.count_nonzero(reasons == "evaluated-relative-natural-residual")
        ),
        "k10_stop_count": int(
            np.count_nonzero(reasons == "evaluated-inner-max-steps")
        ),
        "budget_truncation_count": budget_truncations,
        "budget_truncation_counted_as_converged": False,
    }


def validate_result_iperp_only(root: Path, result: Path) -> dict[str, Any]:
    required = (
        base.COMMON_RESULT_KEYS
        | base.DISCRETE_RESULT_KEYS
        | base.SCALAR_RESULT_KEYS
        | {"stop_reason"}
    )
    forbidden = [
        key
        for key in required
        if "barrier" in key.lower()
        or "densif" in key.lower()
        or "energy" in key.lower()
    ]
    # Mechanism flags containing "energy" are booleans/counters, not physical
    # arrays, and are required to prove that no energy fallback was used.
    allowed_mechanism_flags = {
        "discrete_scalar_lbfgs_energy_backtracking",
        "discrete_scalar_lbfgs_extra_physical_evaluations",
    }
    forbidden = [key for key in forbidden if key not in allowed_mechanism_flags]
    if forbidden:
        raise RuntimeError(f"continuation allowlist contains forbidden data: {forbidden}")
    with np.load(result, allow_pickle=False) as saved:
        missing = sorted(required.difference(saved.files))
        if missing:
            raise RuntimeError(
                "direct continuation result is missing: " + ", ".join(missing)
            )
        common = _validate_common_flexible(saved)
        mechanism = base._validate_scalar(saved, ARM)
    return {
        **common,
        **mechanism,
        "result_fields_accessed": sorted(required),
        "physical_energy_or_barrier_arrays_accessed": False,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    root = args.root.expanduser().resolve()
    command, label = single_run_command(root)
    paths = member_paths(root)
    result_dir = paths["result_dir"]
    audit_path = paths["audit"]
    log_dir = paths["member_log"]
    assert isinstance(result_dir, Path)
    assert isinstance(audit_path, Path)
    assert isinstance(log_dir, Path)
    existing = sorted(
        result_dir.glob(f"ivm_{label}_central_state_ivm_cfg*_budget254.npz")
    )
    companions = sorted(
        result_dir.glob(f"ivm_{label}_central_state_ivm_cfg*_budget254.audit.json")
    )
    if existing or companions or audit_path.exists() or log_dir.exists():
        target = existing[0] if existing else companions[0] if companions else audit_path
        raise FileExistsError(f"refusing to overwrite continuation cell: {target}")

    result_dir.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    wall_seconds = legacy.run_command(
        command, root, log_dir / "stdout.txt", log_dir / "stderr.txt"
    )
    result = legacy.find_result(result_dir, label, base.CANONICAL_BUDGET)
    summary = validate_result_iperp_only(root, result)
    audit = {
        "member_policy_version": MEMBER_POLICY_VERSION,
        "continuation_of": base.MEMBER_POLICY_VERSION,
        "arm": ARM,
        "alpha": base.CANONICAL_ALPHA,
        "eta": base.CANONICAL_ETA,
        "residual_ratio": base.CANONICAL_RATIO,
        "max_evaluated_steps": base.CANONICAL_MAX_EVALUATED_STEPS,
        "lbfgs_history": base.CANONICAL_HISTORY,
        "kkt_atol": base.CANONICAL_KKT_ATOL,
        "absolute_kkt_stop_enabled": False,
        "budget_per_image": base.CANONICAL_BUDGET,
        "replicate": base.CANONICAL_REPLICATE,
        "stop_rule": "R/R0 <= 0.8 OR evaluated accepted steps == 10",
        "reparameterization_policy": "state_curve/per_image from first state",
        "energy_or_objective_backtracking": False,
        "barrier_fields_accessed": False,
        "barrier_computation_or_densification_performed_by_wrapper": False,
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
        "result": str(result),
        "result_sha256": _file_sha256(result),
        "wall_seconds": wall_seconds,
        "scientific_command_sha256": _compact_command_sha256(command),
        "original_registered_direct_command_sha256": (
            ORIGINAL_DIRECT_COMMAND_SHA256
        ),
        "command_routing_only_continuation": True,
        "command": command,
        **summary,
    }
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
