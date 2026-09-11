#!/usr/bin/env python3
"""Run one registered Ala inexact+reparameterization grid cell.

Every member is an isolated clone of the audited 3389-NF one-way run.  Only
fixed alpha, fixed-subproblem relative residual, evaluated joint-step cap,
the true total-internal-NF limit, and output routing differ.  Physical
``--max-step`` remains 1 angstrom; eta remains 1.1; KKT absolute stopping
remains disabled; the existing ``state_curve/per_image`` reparameterization
and recovery transaction is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import run_ivm_v9_discrete_ffrbk_takeover_grid_shim as runtime
import run_ivm_v9_total_internal_nf_budget as total_budget
import run_stage4_uniform_optimized as legacy


MEMBER_POLICY_VERSION = "stage4_ala_discrete_inexact_reparam_grid_v1"
CANONICAL_REPLICATE = "r1"
CANONICAL_TOTAL_INTERNAL_NF_BUDGET = 5000
CANONICAL_COMMAND = (
    "logs/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1/"
    "members/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_"
    "one_way_takeover_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1/"
    "command.json"
)
CANONICAL_COMMAND_SHA256 = (
    "a970b975338c14b8f39cb6b131ccb80fe44f36c7b3801ea18101cca0f54ef93d"
)
ALPHAS = runtime.ALLOWED_ALPHAS
RESIDUAL_RATIOS = runtime.ALLOWED_RESIDUAL_RATIOS
INNER_MAX_STEPS = runtime.INEXACT_GRID_INNER_MAX_STEPS
ROUTING_FLAGS = frozenset(("--output-dir", "--run-label"))
SCIENTIFIC_VALUE_FLAGS = frozenset(
    (
        "--alpha",
        "--budget",
        "--discrete-inner-max-steps",
        "--discrete-residual-ratio",
    )
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_command_sha256(command: List[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _replace_option(command: List[str], flag: str, value: str) -> None:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"expected exactly one valued {flag}")
    command[positions[0] + 1] = value


def _options(command: List[str]) -> Dict[str, str]:
    options: Dict[str, str] = {}
    index = 2
    while index < len(command):
        flag = command[index]
        if not flag.startswith("--"):
            raise RuntimeError(f"unexpected command token: {flag}")
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            if flag in options:
                raise RuntimeError(f"duplicate command option: {flag}")
            options[flag] = command[index + 1]
            index += 2
        else:
            if flag in options:
                raise RuntimeError(f"duplicate command switch: {flag}")
            options[flag] = ""
            index += 1
    return options


def canonical_command(root: Path) -> List[str]:
    path = (root / CANONICAL_COMMAND).resolve()
    if file_sha256(path) != CANONICAL_COMMAND_SHA256:
        raise RuntimeError("audited 3389-NF command hash drifted")
    command = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(command, list) or not all(
        isinstance(token, str) for token in command
    ):
        raise RuntimeError("audited command is not a string vector")
    return command


def _registered_alpha(alpha: Any) -> float:
    value = float(alpha)
    if not math.isfinite(value) or value not in ALPHAS:
        raise ValueError("alpha is not in 1e-1,...,1e6")
    return value


def _registered_ratio(ratio: Any) -> float:
    value = float(ratio)
    if not math.isfinite(value) or value not in RESIDUAL_RATIOS:
        raise ValueError("residual ratio is not registered")
    return value


def _registered_cap(cap: Any) -> int:
    value = int(cap)
    if str(value) != str(cap).strip() or value not in INNER_MAX_STEPS:
        raise ValueError("inner max steps is not registered")
    return value


def alpha_tag(alpha: Any) -> str:
    alpha = _registered_alpha(alpha)
    exponent = int(round(math.log10(alpha)))
    return "a1e" + ("m" + str(-exponent) if exponent < 0 else str(exponent))


def ratio_tag(ratio: Any) -> str:
    ratio = _registered_ratio(ratio)
    return f"rr{int(round(100 * ratio)):02d}"


def cap_tag(cap: Any) -> str:
    return f"c{_registered_cap(cap):02d}"


def run_label(alpha: Any, ratio: Any, cap: Any) -> str:
    return "_".join(
        (
            MEMBER_POLICY_VERSION,
            alpha_tag(alpha),
            ratio_tag(ratio),
            cap_tag(cap),
            "tnf5000",
            CANONICAL_REPLICATE,
        )
    )


def controls(alpha: Any, ratio: Any, cap: Any) -> runtime.GridControls:
    return runtime.validate_grid_controls(
        runtime.GridControls(
            alpha=_registered_alpha(alpha),
            residual_ratio=_registered_ratio(ratio),
            inner_max_steps=_registered_cap(cap),
            total_internal_nf_budget=CANONICAL_TOTAL_INTERNAL_NF_BUDGET,
        )
    )


def assert_registered_command(
    command: List[str], baseline: List[str], cell: runtime.GridControls
) -> None:
    cell = controls(cell.alpha, cell.residual_ratio, cell.inner_max_steps)
    if len(command) != len(baseline) + 2:
        raise RuntimeError("grid command length differs from registered clone")
    if command[-2:] != [
        total_budget.TOTAL_INTERNAL_NF_FLAG,
        str(CANONICAL_TOTAL_INTERNAL_NF_BUDGET),
    ]:
        raise RuntimeError("grid total-NF option is missing or misplaced")
    allowed_indices = {1}
    for flag in ROUTING_FLAGS | SCIENTIFIC_VALUE_FLAGS:
        occurrences = [index for index, token in enumerate(baseline) if token == flag]
        if len(occurrences) != 1:
            raise RuntimeError(f"baseline has invalid {flag}")
        allowed_indices.add(occurrences[0] + 1)
    drift = [
        index
        for index, (left, right) in enumerate(zip(command[:-2], baseline))
        if left != right and index not in allowed_indices
    ]
    if drift:
        raise RuntimeError(
            "grid command changed a frozen token at "
            + ",".join(str(item) for item in drift)
        )
    expected_runtime = str(
        (Path(_options(command)["--root"]) / "src" /
         "run_ivm_v9_discrete_ffrbk_takeover_grid_shim.py").resolve()
    )
    if command[1] != expected_runtime:
        raise RuntimeError("grid command uses the wrong runtime")
    options = _options(command)
    required = {
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
        "--discrete-inner-max-steps": str(cell.inner_max_steps),
        "--discrete-stop-mode": "r08c10",
        "--discrete-residual-ratio": f"{cell.residual_ratio:g}",
        "--discrete-history": "7",
        "--discrete-step-multiplier-eta": "1.1",
        "--plbfgs-kkt-atol": "0",
        "--discrete-ffrbk-selector-mode": "one_way_takeover",
        total_budget.TOTAL_INTERNAL_NF_FLAG: "5000",
    }
    for flag, expected in required.items():
        if options.get(flag) != expected:
            raise RuntimeError(f"grid command changed registered option {flag}")
    if options.get("--state-curve-direct-solve") != "":
        raise RuntimeError("grid command disabled state-curve direct solve")


def single_run_command(
    root: Path, alpha: Any, ratio: Any, cap: Any
) -> Tuple[List[str], str]:
    root = root.expanduser().resolve()
    cell = controls(alpha, ratio, cap)
    baseline = canonical_command(root)
    command = list(baseline)
    command[1] = str(
        (root / "src/run_ivm_v9_discrete_ffrbk_takeover_grid_shim.py").resolve()
    )
    label = run_label(cell.alpha, cell.residual_ratio, cell.inner_max_steps)
    _replace_option(command, "--output-dir", f"runs/{MEMBER_POLICY_VERSION}/members")
    _replace_option(command, "--run-label", label)
    _replace_option(command, "--alpha", f"{cell.alpha:g}")
    _replace_option(command, "--budget", str(CANONICAL_TOTAL_INTERNAL_NF_BUDGET))
    _replace_option(
        command, "--discrete-inner-max-steps", str(cell.inner_max_steps)
    )
    _replace_option(
        command, "--discrete-residual-ratio", f"{cell.residual_ratio:g}"
    )
    command.extend(
        [
            total_budget.TOTAL_INTERNAL_NF_FLAG,
            str(CANONICAL_TOTAL_INTERNAL_NF_BUDGET),
        ]
    )
    assert_registered_command(command, baseline, cell)
    return command, label


def member_paths(
    root: Path, alpha: Any, ratio: Any, cap: Any
) -> Dict[str, Any]:
    label = run_label(alpha, ratio, cap)
    return {
        "label": label,
        "result_dir": root / "runs" / MEMBER_POLICY_VERSION / "members",
        "audit": root / "runs" / MEMBER_POLICY_VERSION / "audits" / f"{label}.json",
        "member_log": root / "logs" / MEMBER_POLICY_VERSION / "members" / label,
    }


def _scalar(saved: np.lib.npyio.NpzFile, key: str):
    return np.asarray(saved[key]).item()


def validate_result(
    result: Path, cell: runtime.GridControls
) -> Dict[str, Any]:
    cell = controls(cell.alpha, cell.residual_ratio, cell.inner_max_steps)
    required = {
        "active_alpha",
        "alpha",
        "alpha_mode",
        "budget",
        "calls_per_image",
        "configuration_sha256",
        "counter_vectors",
        "discrete_all_returned_states_evaluated",
        "discrete_audit_outer",
        "discrete_audit_selection",
        "discrete_audit_step",
        "discrete_audit_stop_reason",
        "discrete_ffrbk_selector_aborted_selection_count",
        "discrete_ffrbk_selector_aborted_solver_calls",
        "discrete_ffrbk_selector_committed_selection_count",
        "discrete_ffrbk_selector_energy_backtracking",
        "discrete_ffrbk_selector_extra_physical_evaluations",
        "discrete_ffrbk_selector_fixed_solver_calls",
        "discrete_ffrbk_selector_inner_max_steps",
        "discrete_ffrbk_selector_kkt_atol",
        "discrete_ffrbk_selector_latch_reset_count",
        "discrete_ffrbk_selector_mode",
        "discrete_ffrbk_selector_residual_ratio",
        "discrete_ffrbk_selector_runtime_policy_id",
        "discrete_ffrbk_selector_uncommitted_selection_count",
        "discrete_history",
        "discrete_inner_max_steps",
        "discrete_residual_ratio",
        "discrete_step_multiplier_aborted_selection_count",
        "discrete_step_multiplier_energy_backtracking",
        "discrete_step_multiplier_eta",
        "discrete_step_multiplier_extra_physical_evaluations",
        "discrete_step_multiplier_uncommitted_selection_count",
        "discrete_stop_mode",
        "frozen_subproblem",
        "i_perp",
        "max_step",
        "plbfgs_kkt_atol",
        "reparam_committed",
        "reparam_every",
        "reparam_mode",
        "segment_length_cv",
        "segment_length_ratio",
        "state_curve_direct_objective_acceptance",
        "state_curve_direct_solve",
        "stop_reason",
        "total_internal_joint_evaluations",
        "total_internal_nf_budget",
        "total_internal_nf_budget_blocked_miss_count",
        "total_internal_nf_budget_exhausted",
        "total_internal_nf_budget_no_overshoot",
        "total_internal_nf_budget_stop_reason",
        "total_internal_nf_complete_state_definition",
        "total_internal_nf_consumed",
        "total_internal_nf_remaining",
        "total_internal_nf_last_complete_state_index",
        "total_internal_nf_last_complete_state_nf",
        "total_internal_nf_terminal_partial_tail_present",
        "total_internal_nf_terminal_partial_tail_quarantined",
        "total_internal_nf_terminal_record_is_complete_outer_state",
        "total_internal_nf_terminal_uncommitted_charged_tail_nf",
    }
    with np.load(result, allow_pickle=False) as saved:
        missing = sorted(required.difference(saved.files))
        if missing:
            raise RuntimeError("grid result is missing: " + ", ".join(missing))
        exact_scalars = {
            "alpha": cell.alpha,
            "alpha_mode": "fixed",
            "budget": CANONICAL_TOTAL_INTERNAL_NF_BUDGET,
            "discrete_ffrbk_selector_inner_max_steps": cell.inner_max_steps,
            "discrete_ffrbk_selector_kkt_atol": 0.0,
            "discrete_ffrbk_selector_mode": "one_way_takeover",
            "discrete_ffrbk_selector_residual_ratio": cell.residual_ratio,
            "discrete_ffrbk_selector_runtime_policy_id": runtime.POLICY_ID,
            "discrete_history": 7,
            "discrete_inner_max_steps": cell.inner_max_steps,
            "discrete_residual_ratio": cell.residual_ratio,
            "discrete_step_multiplier_eta": 1.1,
            "discrete_stop_mode": "r08c10",
            "frozen_subproblem": "discrete_b040",
            "max_step": 1.0,
            "plbfgs_kkt_atol": 0.0,
            "reparam_every": 1,
            "reparam_mode": "state_curve",
            "state_curve_direct_objective_acceptance": "per_image",
            "state_curve_direct_solve": True,
            "total_internal_nf_budget": CANONICAL_TOTAL_INTERNAL_NF_BUDGET,
            "total_internal_nf_budget_no_overshoot": True,
            "total_internal_nf_complete_state_definition": (
                total_budget.COMPLETENESS_INITIAL_OR_REPARAM_COMMITTED
            ),
        }
        for key, expected in exact_scalars.items():
            if _scalar(saved, key) != expected:
                raise RuntimeError(f"grid result drifted at {key}")
        active_alpha = np.asarray(saved["active_alpha"], dtype=float)
        if not active_alpha.size or not np.all(active_alpha == cell.alpha):
            raise RuntimeError("grid alpha was not fixed throughout")
        for key in (
            "discrete_ffrbk_selector_energy_backtracking",
            "discrete_step_multiplier_energy_backtracking",
        ):
            if bool(_scalar(saved, key)):
                raise RuntimeError("grid runtime enabled energy backtracking")
        for key in (
            "discrete_ffrbk_selector_extra_physical_evaluations",
            "discrete_ffrbk_selector_uncommitted_selection_count",
            "discrete_step_multiplier_extra_physical_evaluations",
            "discrete_step_multiplier_uncommitted_selection_count",
        ):
            if int(_scalar(saved, key)) != 0:
                raise RuntimeError(f"grid result has forbidden count at {key}")
        if not bool(_scalar(saved, "discrete_all_returned_states_evaluated")):
            raise RuntimeError("grid result contains an unevaluated returned state")

        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=np.int64)
        calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        mesh_ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        shapes = {item.shape for item in (cost, calls, committed, i_perp, cv, mesh_ratio)}
        if len(shapes) != 1 or cost.ndim != 1 or cost.size < 1:
            raise RuntimeError("grid trajectory arrays are misaligned")
        if counters.shape != (cost.size, 22):
            raise RuntimeError("grid counter-vector shape is invalid")
        if not np.array_equal(cost, np.sum(counters[:, 1:-1], axis=1)):
            raise RuntimeError("grid total NF differs from counter vectors")
        if not np.array_equal(calls, np.max(counters[:, 1:-1], axis=1)):
            raise RuntimeError("grid per-image call labels differ from counters")
        if np.any(np.diff(cost) <= 0) or np.any(cost > 5000):
            raise RuntimeError("grid total NF is not strict or overshot")
        if not (
            np.all(np.isfinite(i_perp))
            and np.all(np.isfinite(cv))
            and np.all(np.isfinite(mesh_ratio))
            and np.all(i_perp >= 0.0)
            and np.all(cv >= 0.0)
            and np.all(mesh_ratio >= 1.0)
        ):
            raise RuntimeError("grid I_perp or mesh diagnostics are invalid")
        consumed = int(_scalar(saved, "total_internal_nf_consumed"))
        exhausted = bool(_scalar(saved, "total_internal_nf_budget_exhausted"))
        if consumed != int(cost[-1]) or exhausted != (consumed == 5000):
            raise RuntimeError("grid total-budget terminal metadata is inconsistent")
        remaining = int(_scalar(saved, "total_internal_nf_remaining"))
        if remaining != 5000 - consumed or remaining < 0:
            raise RuntimeError("grid total-budget remaining metadata is inconsistent")
        budget_stop_reason = str(
            _scalar(saved, "total_internal_nf_budget_stop_reason")
        )
        allowed_budget_stop_reasons = {
            "total_internal_joint_budget_exhausted",
            "total_internal_joint_budget_insufficient_for_full_band",
            "runner_stopped_before_total_internal_joint_budget",
        }
        if budget_stop_reason not in allowed_budget_stop_reasons:
            raise RuntimeError("grid total-budget stop reason is unknown")
        if exhausted != (
            budget_stop_reason == "total_internal_joint_budget_exhausted"
        ):
            raise RuntimeError("grid total-budget stop reason is inconsistent")

        complete = committed.copy()
        complete[0] = True
        last_complete_index = int(np.flatnonzero(complete)[-1])
        last_complete_nf = int(cost[last_complete_index])
        terminal_complete = bool(complete[-1])
        terminal_tail = consumed - last_complete_nf
        partial_tail = bool(not terminal_complete and terminal_tail > 0)
        metadata_checks = {
            "total_internal_nf_last_complete_state_index": last_complete_index,
            "total_internal_nf_last_complete_state_nf": last_complete_nf,
            "total_internal_nf_terminal_record_is_complete_outer_state": terminal_complete,
            "total_internal_nf_terminal_uncommitted_charged_tail_nf": terminal_tail,
            "total_internal_nf_terminal_partial_tail_present": partial_tail,
            "total_internal_nf_terminal_partial_tail_quarantined": partial_tail,
        }
        for key, expected in metadata_checks.items():
            if _scalar(saved, key) != expected:
                raise RuntimeError(f"grid terminal-tail metadata drifted at {key}")

        selection = np.asarray(saved["discrete_audit_selection"], dtype="U8")
        outer = np.asarray(saved["discrete_audit_outer"], dtype=int)
        step = np.asarray(saved["discrete_audit_step"], dtype=int)
        if selection.shape != outer.shape or selection.shape != step.shape:
            raise RuntimeError("grid selector audit arrays are misaligned")
        if np.any(~np.isin(selection, ["H0", "LBFGS"])):
            raise RuntimeError("grid selector audit has an unknown direction")
        if np.any(step < 0) or np.any(step >= cell.inner_max_steps):
            raise RuntimeError("grid fixed solve exceeded its inner cap")
        for outer_index in np.unique(outer):
            observed = step[outer == outer_index]
            if not np.array_equal(observed, np.arange(observed.size)):
                raise RuntimeError("grid fixed-solve steps are not contiguous")
        fixed_calls = int(_scalar(saved, "discrete_ffrbk_selector_fixed_solver_calls"))
        aborted_solvers = int(_scalar(saved, "discrete_ffrbk_selector_aborted_solver_calls"))
        aborted_selector = int(_scalar(saved, "discrete_ffrbk_selector_aborted_selection_count"))
        aborted_eta = int(_scalar(saved, "discrete_step_multiplier_aborted_selection_count"))
        if int(_scalar(saved, "discrete_ffrbk_selector_latch_reset_count")) != fixed_calls:
            raise RuntimeError("grid one-way latch did not reset per fixed solve")
        if int(_scalar(saved, "discrete_ffrbk_selector_committed_selection_count")) != selection.size:
            raise RuntimeError("grid selector committed count is misaligned")
        if aborted_solvers not in (0, 1) or aborted_selector != aborted_eta:
            raise RuntimeError("grid terminal aborted-selector audit is invalid")
        if (aborted_solvers or aborted_selector) and not exhausted:
            raise RuntimeError("grid selector aborted before total-budget exhaustion")

        mesh_eligible = complete & (cv <= 0.0065) & (mesh_ratio <= 1.03)
        return {
            "configuration_sha256": str(_scalar(saved, "configuration_sha256")),
            "trajectory_record_count": int(cost.size),
            "final_optimization_NF": consumed,
            "total_budget_exhausted": exhausted,
            "total_budget_stop_reason": budget_stop_reason,
            "total_budget_blocked_miss_count": int(
                _scalar(saved, "total_internal_nf_budget_blocked_miss_count")
            ),
            "complete_state_count": int(np.count_nonzero(complete)),
            "last_complete_state_NF": last_complete_nf,
            "terminal_partial_tail_NF": terminal_tail,
            "terminal_partial_tail_quarantined": partial_tail,
            "mesh_eligible_complete_state_count": int(np.count_nonzero(mesh_eligible)),
            "fixed_solver_call_count": fixed_calls,
            "discrete_joint_step_count": int(selection.size),
            "h0_selection_count": int(np.count_nonzero(selection == "H0")),
            "lbfgs_selection_count": int(np.count_nonzero(selection == "LBFGS")),
            "aborted_solver_count": aborted_solvers,
            "physical_stop_reason": str(_scalar(saved, "stop_reason")),
            "result_fields_accessed": sorted(required),
        }


def validate_companion(result: Path, final_internal_nf: int, blocked: int) -> Dict[str, Any]:
    path = result.with_name(result.stem + ".audit.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_image = np.asarray(payload["per_image_joint_evaluations"], dtype=int)
    blocked_per_image = np.asarray(payload["per_image_budget_blocked_misses"], dtype=int)
    if per_image.shape != (22,) or blocked_per_image.shape != (22,):
        raise RuntimeError("grid companion counter shapes are invalid")
    if int(np.sum(per_image[1:-1])) != final_internal_nf:
        raise RuntimeError("grid companion internal NF differs from result")
    if per_image[0] != 1 or per_image[-1] != 1:
        raise RuntimeError("grid endpoint convention drifted")
    if int(np.sum(blocked_per_image[1:-1])) != blocked:
        raise RuntimeError("grid blocked-miss audit differs from runtime metadata")
    if not bool(payload["consistent"]):
        raise RuntimeError("grid backend/evaluator counter audit is inconsistent")
    if int(payload["backend_raw_joint_evaluations"]) != final_internal_nf + 2:
        raise RuntimeError("grid backend raw count is not internal NF plus endpoints")
    if int(payload["sum_image_joint_evaluations"]) != final_internal_nf + 2:
        raise RuntimeError("grid image counter sum is not internal NF plus endpoints")
    return {
        "companion_audit": str(path.resolve()),
        "companion_audit_sha256": file_sha256(path),
        "backend_raw_joint_evaluations": int(payload["backend_raw_joint_evaluations"]),
        "endpoint_joint_evaluations": [int(per_image[0]), int(per_image[-1])],
        "counter_consistent": True,
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--residual-ratio", required=True)
    parser.add_argument("--inner-max-steps", required=True)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    root = args.root.expanduser().resolve()
    cell = controls(args.alpha, args.residual_ratio, args.inner_max_steps)
    command, label = single_run_command(
        root, cell.alpha, cell.residual_ratio, cell.inner_max_steps
    )
    paths = member_paths(root, cell.alpha, cell.residual_ratio, cell.inner_max_steps)
    result_dir = paths["result_dir"]
    audit_path = paths["audit"]
    log_dir = paths["member_log"]
    existing = sorted(
        result_dir.glob(f"ivm_{label}_central_state_ivm_cfg*_budget5000.npz")
    ) + sorted(
        result_dir.glob(
            f"ivm_{label}_central_state_ivm_cfg*_budget5000.audit.json"
        )
    )
    if existing or audit_path.exists() or log_dir.exists():
        target = existing[0] if existing else audit_path if audit_path.exists() else log_dir
        raise FileExistsError(f"refusing to overwrite grid cell: {target}")
    result_dir.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    wall_seconds = legacy.run_command(
        command, root, log_dir / "stdout.txt", log_dir / "stderr.txt"
    )
    result = legacy.find_result(result_dir, label, 5000)
    summary = validate_result(result, cell)
    companion = validate_companion(
        result,
        int(summary["final_optimization_NF"]),
        int(summary["total_budget_blocked_miss_count"]),
    )
    audit = {
        "member_policy_version": MEMBER_POLICY_VERSION,
        "alpha": cell.alpha,
        "residual_ratio": cell.residual_ratio,
        "inner_max_steps": cell.inner_max_steps,
        "stop_rule": (
            f"R/R0 <= {cell.residual_ratio:g} OR fully evaluated joint steps "
            f"== {cell.inner_max_steps}"
        ),
        "physical_max_step_angstrom": 1.0,
        "eta": 1.1,
        "kkt_atol": 0.0,
        "lbfgs_history": 7,
        "total_internal_nf_budget": 5000,
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
