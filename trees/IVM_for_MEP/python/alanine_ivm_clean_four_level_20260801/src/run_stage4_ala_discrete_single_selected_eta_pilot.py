#!/usr/bin/env python3
"""One preregistered selected-step eta pilot member for Ala 66D.

The canonical alpha=12.5, r=0.8, kmax=10, N254 discrete candidate command is
cloned exactly.  Only the entry point (the runtime eta shim), output routing,
and the one registered post-selection multiplier differ.  The wrapper reads
only geometry, I_perp, cost, mesh, mechanism, and eta-audit fields; all energy
barrier fields remain embargoed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import run_ivm_v9_discrete_selected_eta_shim as eta_shim
import run_stage4_ala_discrete_single_alpha_refinement as frozen_refinement
import run_stage4_ala_discrete_single_fixedalpha as coarse_single
import run_stage4_uniform_optimized as legacy


MEMBER_POLICY_VERSION = "stage4_ala_discrete_single_selected_eta_pilot_v1"
ETA_VALUES = eta_shim.PILOT_ETA_VALUES
ETA_TEXT = {1.0: "1", 1.25: "1.25", 1.5: "1.5", 2.0: "2"}
ETA_TAG = {1.0: "eta1", 1.25: "eta1p25", 1.5: "eta1p5", 2.0: "eta2"}
CANONICAL_ALPHA = 12.5
CANONICAL_RATIO = 0.8
CANONICAL_MAX_EVALUATED_STEPS = 10
CANONICAL_HISTORY = 7
CANONICAL_BUDGET = 254
CANONICAL_REPLICATE = "r1"
MESH_CV_MAX = 0.0065
MESH_RATIO_MAX = 1.03
ROUTING_OPTIONS = frozenset({"--output-dir", "--run-label"})
ETA_ONE_ORACLE = (
    "runs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/members/"
    "ivm_stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "r08c10_a12p5_N254_discrete_b040_r1_central_state_ivm_"
    "cfgfaf71a0dca2c_budget254.npz"
)
ETA_ONE_ORACLE_SHA256 = (
    "15df24a92b0a4a73e972be0574b5979d2e4eb7e7b980bc76cd58d474f9d8e835"
)

COMMON_RESULT_KEYS = frozenset(
    {
        "active_alpha",
        "alpha",
        "alpha_mode",
        "configuration_sha256",
        "i_perp",
        "initial_path_sha256",
        "physical_fmax",
        "reparam_committed",
        "reparam_mode",
        "segment_length_cv",
        "segment_length_ratio",
        "state_curve_direct_objective_acceptance",
        "stop_complete_iperp",
        "stored_path_calls",
        "stored_paths",
        "subproblem_line_search_backtracks",
        "total_internal_joint_evaluations",
    }
)
DISCRETE_RESULT_KEYS = frozenset(
    {
        "discrete_all_returned_states_evaluated",
        "discrete_audit_curvature_status",
        "discrete_audit_global_relative_residual",
        "discrete_audit_h0_cap_scale",
        "discrete_audit_h0_capped_predicted_decrease",
        "discrete_audit_h0_raw_predicted_decrease",
        "discrete_audit_history_count",
        "discrete_audit_lbfgs_cap_scale",
        "discrete_audit_lbfgs_capped_predicted_decrease",
        "discrete_audit_lbfgs_raw_predicted_decrease",
        "discrete_audit_outer",
        "discrete_audit_selection",
        "discrete_audit_step",
        "discrete_audit_stop_reason",
        "discrete_h0_selection_count",
        "discrete_history",
        "discrete_implementation_id",
        "discrete_inner_max_steps",
        "discrete_lbfgs_selection_count",
        "discrete_residual_ratio",
        "discrete_stop_mode",
        "frozen_subproblem",
    }
)
ETA_RESULT_KEYS = frozenset(
    {
        "discrete_step_multiplier_policy_id",
        "discrete_step_multiplier_base_implementation_id",
        "discrete_step_multiplier_implementation_id",
        "discrete_step_multiplier_eta",
        "discrete_step_multiplier_application_stage",
        "discrete_step_multiplier_selector_isolated",
        "discrete_step_multiplier_dynamic_cap_reapplied",
        "discrete_step_multiplier_energy_backtracking",
        "discrete_step_multiplier_extra_physical_evaluations",
        "discrete_step_multiplier_audit_selection",
        "discrete_step_multiplier_audit_requested",
        "discrete_step_multiplier_audit_applied",
        "discrete_step_multiplier_audit_cap_hit",
        "discrete_step_multiplier_audit_dynamic_max_step",
        "discrete_step_multiplier_audit_base_max_atomic_displacement",
        "discrete_step_multiplier_audit_requested_max_atomic_displacement",
        "discrete_step_multiplier_audit_applied_max_atomic_displacement",
        "discrete_step_multiplier_uncommitted_selection_count",
        "discrete_step_multiplier_aborted_selection_count",
    }
)
ETA_ONE_PARITY_KEYS = frozenset(
    {
        "active_alpha",
        "i_perp",
        "physical_fmax",
        "reparam_committed",
        "segment_length_cv",
        "segment_length_ratio",
        "stored_path_calls",
        "stored_paths",
        "subproblem_line_search_backtracks",
        "total_internal_joint_evaluations",
        "discrete_all_returned_states_evaluated",
        "discrete_audit_curvature_status",
        "discrete_audit_global_relative_residual",
        "discrete_audit_h0_cap_scale",
        "discrete_audit_h0_capped_predicted_decrease",
        "discrete_audit_h0_raw_predicted_decrease",
        "discrete_audit_history_count",
        "discrete_audit_lbfgs_cap_scale",
        "discrete_audit_lbfgs_capped_predicted_decrease",
        "discrete_audit_lbfgs_raw_predicted_decrease",
        "discrete_audit_outer",
        "discrete_audit_selection",
        "discrete_audit_step",
        "discrete_audit_stop_reason",
        "discrete_h0_selection_count",
        "discrete_lbfgs_selection_count",
    }
)


def _require_eta(value: float) -> float:
    eta = eta_shim.validate_eta(value)
    if eta not in ETA_VALUES:
        raise ValueError("eta must be one of 1,1.25,1.5,2")
    return eta


def eta_tag(value: float) -> str:
    return ETA_TAG[_require_eta(value)]


def run_label(value: float) -> str:
    eta = _require_eta(value)
    return (
        f"{MEMBER_POLICY_VERSION}_{eta_tag(eta)}_r08c10_a12p5_"
        f"N254_discrete_b040_{CANONICAL_REPLICATE}"
    )


def _replace_option(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"Expected one valued {flag} option")
    command[positions[0] + 1] = value


def canonical_reference_command(root: Path) -> list[str]:
    command, _ = frozen_refinement.single_run_command(
        root,
        CANONICAL_ALPHA,
        CANONICAL_BUDGET,
        CANONICAL_REPLICATE,
        "candidate",
    )
    return command


def assert_eta_only_scientific_change(
    command: list[str], canonical: list[str], value: float, root: Path
) -> None:
    eta = _require_eta(value)
    if len(command) != len(canonical) + 2:
        raise RuntimeError("Eta command length differs by more than its one option")
    expected = list(canonical)
    expected[1] = str(
        (root / "src/run_ivm_v9_discrete_selected_eta_shim.py").resolve()
    )
    _replace_option(
        expected,
        "--output-dir",
        f"runs/{MEMBER_POLICY_VERSION}/members",
    )
    _replace_option(expected, "--run-label", run_label(eta))
    expected.extend([eta_shim.ETA_FLAG, ETA_TEXT[eta]])
    if command != expected:
        drift = [
            index
            for index, (observed, registered) in enumerate(
                zip(command, expected)
            )
            if observed != registered
        ]
        if len(command) != len(expected):
            drift.append(min(len(command), len(expected)))
        raise RuntimeError(
            "Eta member drifted from the exact cloned command at indices "
            + ",".join(str(index) for index in drift)
        )
    options = coarse_single.command_options(command)
    required = {
        "--alpha": "12.5",
        "--alpha-mode": "fixed",
        "--alpha-min": "0.01",
        "--alpha-max": "inf",
        "--max-step": "1",
        "--history": "7",
        "--budget": "254",
        "--state-globalization": "iperp_monotone",
        "--state-trust-fraction": "2",
        "--reparam-mode": "state_curve",
        "--state-curve-direct-objective-acceptance": "per_image",
        "--stop-complete-iperp": "-1",
        "--frozen-subproblem": "discrete_b040",
        "--discrete-inner-max-steps": "10",
        "--discrete-stop-mode": "r08c10",
        "--discrete-residual-ratio": "0.8",
        "--discrete-history": "7",
        eta_shim.ETA_FLAG: ETA_TEXT[eta],
    }
    for flag, expected_value in required.items():
        if options.get(flag) != expected_value:
            raise RuntimeError(f"Eta member changed frozen option {flag}")


def single_run_command(root: Path, value: float) -> tuple[list[str], str]:
    eta = _require_eta(value)
    canonical = canonical_reference_command(root)
    command = list(canonical)
    command[1] = str(
        (root / "src/run_ivm_v9_discrete_selected_eta_shim.py").resolve()
    )
    label = run_label(eta)
    _replace_option(
        command,
        "--output-dir",
        f"runs/{MEMBER_POLICY_VERSION}/members",
    )
    _replace_option(command, "--run-label", label)
    command.extend([eta_shim.ETA_FLAG, ETA_TEXT[eta]])
    assert_eta_only_scientific_change(command, canonical, eta, root)
    return command, label


def compact_command_sha256(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def member_paths(root: Path, value: float) -> dict[str, Path | str]:
    label = run_label(value)
    return {
        "label": label,
        "result_dir": root / "runs" / MEMBER_POLICY_VERSION / "members",
        "audit": root / "runs" / MEMBER_POLICY_VERSION / "audits" / f"{label}.json",
        "member_log": root / "logs" / MEMBER_POLICY_VERSION / "members" / label,
        "launcher_log": root / "logs" / MEMBER_POLICY_VERSION / "launcher" / label,
    }


def _scalar(saved: np.lib.npyio.NpzFile, key: str):
    return np.asarray(saved[key]).item()


def _require_keys(saved: np.lib.npyio.NpzFile, result: Path) -> None:
    required = COMMON_RESULT_KEYS | DISCRETE_RESULT_KEYS | ETA_RESULT_KEYS
    missing = sorted(required.difference(saved.files))
    if missing:
        raise RuntimeError(f"{result} is missing: {', '.join(missing)}")


def _assert_eta_one_parity(
    root: Path, saved: np.lib.npyio.NpzFile
) -> list[str]:
    oracle = (root / ETA_ONE_ORACLE).resolve()
    if legacy.file_sha256(oracle) != ETA_ONE_ORACLE_SHA256:
        raise RuntimeError("Frozen eta=1 oracle hash drifted")
    with np.load(oracle, allow_pickle=False) as reference:
        missing = sorted(ETA_ONE_PARITY_KEYS.difference(reference.files))
        if missing:
            raise RuntimeError("Eta=1 oracle lacks parity fields")
        for key in sorted(ETA_ONE_PARITY_KEYS):
            if not np.array_equal(np.asarray(saved[key]), np.asarray(reference[key])):
                raise RuntimeError(f"Eta=1 whole-run parity failed at {key}")
    return sorted(ETA_ONE_PARITY_KEYS)


def _validate_result_iperp_only(
    root: Path, result: Path, value: float
) -> dict[str, object]:
    eta = _require_eta(value)
    with np.load(result, allow_pickle=False) as saved:
        _require_keys(saved, result)
        if float(_scalar(saved, "alpha")) != CANONICAL_ALPHA:
            raise RuntimeError("Eta result changed alpha")
        if str(_scalar(saved, "alpha_mode")) != "fixed":
            raise RuntimeError("Eta result changed alpha mode")
        active_alpha = np.asarray(saved["active_alpha"], dtype=float)
        if not active_alpha.size or not np.all(active_alpha == CANONICAL_ALPHA):
            raise RuntimeError("Eta result changed alpha within the trajectory")
        if str(_scalar(saved, "frozen_subproblem")) != "discrete_b040":
            raise RuntimeError("Eta result changed frozen subproblem")
        if str(_scalar(saved, "discrete_stop_mode")) != "r08c10":
            raise RuntimeError("Eta result changed stop mode")
        if int(_scalar(saved, "discrete_inner_max_steps")) != 10:
            raise RuntimeError("Eta result changed evaluated-step cap")
        if float(_scalar(saved, "discrete_residual_ratio")) != 0.8:
            raise RuntimeError("Eta result changed residual ratio")
        if int(_scalar(saved, "discrete_history")) != 7:
            raise RuntimeError("Eta result changed L-BFGS history")
        if str(_scalar(saved, "reparam_mode")) != "state_curve" or str(
            _scalar(saved, "state_curve_direct_objective_acceptance")
        ) != "per_image":
            raise RuntimeError("Eta result changed Ala reparameterization")
        if float(_scalar(saved, "stop_complete_iperp")) != -1.0:
            raise RuntimeError("Eta result used I_perp early stopping")
        if not bool(_scalar(saved, "discrete_all_returned_states_evaluated")):
            raise RuntimeError("Eta result returned an unevaluated state")

        expected_id = eta_shim.shim_implementation_id(
            coarse_single.EXPECTED_IMPLEMENTATION_ID, eta
        )
        if str(_scalar(saved, "discrete_implementation_id")) != expected_id:
            raise RuntimeError("Eta result used the wrong discrete implementation")
        if str(_scalar(saved, "discrete_step_multiplier_policy_id")) != (
            eta_shim.SHIM_POLICY_ID
        ):
            raise RuntimeError("Eta result used the wrong shim policy")
        if str(
            _scalar(saved, "discrete_step_multiplier_base_implementation_id")
        ) != coarse_single.EXPECTED_IMPLEMENTATION_ID:
            raise RuntimeError("Eta result used the wrong base implementation")
        if str(
            _scalar(saved, "discrete_step_multiplier_implementation_id")
        ) != expected_id:
            raise RuntimeError("Eta metadata implementation ID differs")
        if float(_scalar(saved, "discrete_step_multiplier_eta")) != eta:
            raise RuntimeError("Eta metadata carries the wrong multiplier")
        if str(
            _scalar(saved, "discrete_step_multiplier_application_stage")
        ) != eta_shim.APPLICATION_STAGE:
            raise RuntimeError("Eta was applied at the wrong stage")
        if not bool(_scalar(saved, "discrete_step_multiplier_selector_isolated")):
            raise RuntimeError("Eta contaminated the direction selector")
        if not bool(
            _scalar(saved, "discrete_step_multiplier_dynamic_cap_reapplied")
        ):
            raise RuntimeError("Eta did not reapply the dynamic cap")
        if bool(_scalar(saved, "discrete_step_multiplier_energy_backtracking")):
            raise RuntimeError("Eta run enabled forbidden energy backtracking")
        if int(
            _scalar(saved, "discrete_step_multiplier_extra_physical_evaluations")
        ) != 0:
            raise RuntimeError("Eta shim charged extra physical evaluations")
        if int(
            _scalar(saved, "discrete_step_multiplier_aborted_selection_count")
        ) != 0:
            raise RuntimeError("Eta run contains aborted selector events")

        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        mesh_ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        shapes = {array.shape for array in (cost, i_perp, committed, cv, mesh_ratio)}
        if len(shapes) != 1 or cost.ndim != 1 or cost.size != CANONICAL_BUDGET:
            raise RuntimeError("Eta trajectory arrays have invalid shapes")
        if np.any(np.diff(cost) <= 0):
            raise RuntimeError("Eta optimization NF is not strictly increasing")
        if not all(
            np.all(np.isfinite(array)) for array in (i_perp, cv, mesh_ratio)
        ):
            raise RuntimeError("Eta I_perp or mesh data are non-finite")
        if np.any(i_perp < 0.0) or np.any(cv < 0.0) or np.any(mesh_ratio < 1.0):
            raise RuntimeError("Eta I_perp or mesh data are invalid")
        if np.any(np.asarray(saved["subproblem_line_search_backtracks"], int) != 0):
            raise RuntimeError("Eta discrete run recorded forbidden backtracking")

        selection = np.asarray(saved["discrete_audit_selection"], dtype="U8")
        eta_selection = np.asarray(
            saved["discrete_step_multiplier_audit_selection"], dtype="U8"
        )
        requested = np.asarray(
            saved["discrete_step_multiplier_audit_requested"], dtype=float
        )
        applied = np.asarray(
            saved["discrete_step_multiplier_audit_applied"], dtype=float
        )
        cap_hit = np.asarray(
            saved["discrete_step_multiplier_audit_cap_hit"], dtype=bool
        )
        dynamic = np.asarray(
            saved["discrete_step_multiplier_audit_dynamic_max_step"], dtype=float
        )
        base_max = np.asarray(
            saved[
                "discrete_step_multiplier_audit_base_max_atomic_displacement"
            ],
            dtype=float,
        )
        requested_max = np.asarray(
            saved[
                "discrete_step_multiplier_audit_requested_max_atomic_displacement"
            ],
            dtype=float,
        )
        applied_max = np.asarray(
            saved[
                "discrete_step_multiplier_audit_applied_max_atomic_displacement"
            ],
            dtype=float,
        )
        audit_arrays = (
            eta_selection,
            requested,
            applied,
            cap_hit,
            dynamic,
            base_max,
            requested_max,
            applied_max,
        )
        if any(array.shape != selection.shape for array in audit_arrays):
            raise RuntimeError("Eta audit arrays do not align with selections")
        if not np.array_equal(selection, eta_selection):
            raise RuntimeError("Eta audit changed selected direction labels")
        if not np.all(requested == eta):
            raise RuntimeError("Eta audit requested multiplier drifted")
        if not all(
            np.all(np.isfinite(array))
            for array in (applied, dynamic, base_max, requested_max, applied_max)
        ):
            raise RuntimeError("Eta audit contains non-finite geometry")
        if np.any(applied <= 0.0) or np.any(applied > eta * (1.0 + 1e-12)):
            raise RuntimeError("Eta applied multiplier is invalid")
        if np.any(dynamic <= 0.0) or np.any(base_max <= 0.0):
            raise RuntimeError("Eta dynamic cap/base displacement is invalid")
        if not np.allclose(requested_max, eta * base_max, rtol=1e-13, atol=1e-14):
            raise RuntimeError("Eta requested displacement is not eta times base")
        if np.any(applied_max > dynamic * (1.0 + 1e-12)):
            raise RuntimeError("Eta applied displacement exceeds dynamic cap")
        if not np.allclose(applied, applied_max / base_max, rtol=1e-12, atol=1e-14):
            raise RuntimeError("Eta applied multiplier/displacement mismatch")
        if not np.array_equal(cap_hit, requested_max > dynamic):
            raise RuntimeError("Eta cap-hit audit is inconsistent")
        if eta == 1.0 and (
            np.any(cap_hit)
            or not np.all(applied == 1.0)
            or not np.array_equal(base_max, requested_max)
            or not np.array_equal(base_max, applied_max)
        ):
            raise RuntimeError("Eta=1 violated its exact selected-step identity")

        h0 = int(np.count_nonzero(selection == "H0"))
        lbfgs = int(np.count_nonzero(selection == "LBFGS"))
        if h0 != int(_scalar(saved, "discrete_h0_selection_count")) or lbfgs != int(
            _scalar(saved, "discrete_lbfgs_selection_count")
        ):
            raise RuntimeError("Eta direction counts differ from selection audit")

        eta_one_parity_keys: list[str] = []
        if eta == 1.0:
            eta_one_parity_keys = _assert_eta_one_parity(root, saved)

        complete = committed.copy()
        complete[0] = True
        eligible = complete & (cv <= MESH_CV_MAX) & (mesh_ratio <= MESH_RATIO_MAX)
        accessed = COMMON_RESULT_KEYS | DISCRETE_RESULT_KEYS | ETA_RESULT_KEYS
        if any("barrier" in key.lower() for key in accessed):
            raise RuntimeError("Eta validator allowlist contains a barrier field")
        return {
            "configuration_sha256": str(_scalar(saved, "configuration_sha256")),
            "initial_path_sha256": str(_scalar(saved, "initial_path_sha256")),
            "trajectory_record_count": int(cost.size),
            "final_optimization_NF": int(cost[-1]),
            "complete_state_definition": "initial_plus_reparam_committed",
            "complete_state_NF": cost[complete].tolist(),
            "mesh_eligibility": "CV<=0.0065 and ratio<=1.03",
            "mesh_eligible_complete_state_NF": cost[eligible].tolist(),
            "mesh_eligible_complete_state_count": int(np.count_nonzero(eligible)),
            "h0_selection_count": h0,
            "lbfgs_selection_count": lbfgs,
            "eta_cap_hit_count": int(np.count_nonzero(cap_hit)),
            "eta_applied_multiplier_min": float(np.min(applied)),
            "eta_applied_multiplier_max": float(np.max(applied)),
            "eta_uncommitted_selection_count": int(
                _scalar(saved, "discrete_step_multiplier_uncommitted_selection_count")
            ),
            "eta_one_whole_run_parity_or_not_applicable": eta == 1.0,
            "eta_one_parity_keys": eta_one_parity_keys,
            "result_fields_accessed": sorted(accessed),
        }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--eta", type=float, choices=ETA_VALUES, required=True)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    root = args.root.expanduser().resolve()
    eta = _require_eta(args.eta)
    command, label = single_run_command(root, eta)
    paths = member_paths(root, eta)
    result_dir, audit_path, log_dir = (
        paths["result_dir"],
        paths["audit"],
        paths["member_log"],
    )
    assert isinstance(result_dir, Path)
    assert isinstance(audit_path, Path)
    assert isinstance(log_dir, Path)
    existing = sorted(
        result_dir.glob(f"ivm_{label}_central_state_ivm_cfg*_budget254.npz")
    )
    if existing or audit_path.exists() or log_dir.exists():
        target = existing[0] if existing else audit_path
        raise FileExistsError(f"Refusing to overwrite eta member: {target}")
    result_dir.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    wall_seconds = legacy.run_command(
        command, root, log_dir / "stdout.txt", log_dir / "stderr.txt"
    )
    result = legacy.find_result(result_dir, label, CANONICAL_BUDGET)
    summary = _validate_result_iperp_only(root, result, eta)
    audit: dict[str, object] = {
        "member_policy_version": MEMBER_POLICY_VERSION,
        "member_role": "same_build_eta1_control" if eta == 1.0 else "eta_candidate",
        "eta": eta,
        "eta_policy": eta_shim.SHIM_POLICY_ID,
        "eta_application_stage": eta_shim.APPLICATION_STAGE,
        "alpha": CANONICAL_ALPHA,
        "fixed_alpha_throughout": True,
        "residual_ratio": CANONICAL_RATIO,
        "max_evaluated_steps": CANONICAL_MAX_EVALUATED_STEPS,
        "lbfgs_history": CANONICAL_HISTORY,
        "budget_per_image": CANONICAL_BUDGET,
        "replicate": CANONICAL_REPLICATE,
        "scientific_change_from_frozen_candidate": "only post-selection eta",
        "runner_change_is_runtime_shim_only": True,
        "selector_isolated_from_eta": True,
        "same_dynamic_cap_reapplied": True,
        "energy_backtracking": False,
        "extra_physical_evaluations_from_shim": 0,
        "stop_rule": "R/R0 <= 0.8 OR evaluated accepted steps == 10",
        "reparameterization_policy": "state_curve/per_image from first state",
        "single_run_from_canonical_common_path": True,
        "iperp_only_scientific_analysis": True,
        "barrier_fields_accessed": False,
        "barrier_computation_or_densification_performed_by_wrapper": False,
        "formal_ranking_deferred_to_complete_mesh_iperp_ctp": True,
        "result": str(result),
        "result_sha256": legacy.file_sha256(result),
        "wall_seconds": wall_seconds,
        "scientific_command_sha256": compact_command_sha256(command),
        "command": command,
        **summary,
    }
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
