#!/usr/bin/env python3
"""One isolated fixed eta=1.1 follow-up member for Ala 66D.

Relative to the formal ``stage4_ala_discrete_single_fixedalpha_v2``
alpha=20/r=0.8/kmax=10/N254 candidate, the only
scientific change is the post-selection selected-step multiplier eta=1.1.
The already locked runtime shim is used without modification.  This wrapper
never indexes an energy-barrier field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import run_ivm_v9_discrete_selected_eta_shim as eta_shim
import run_stage4_ala_discrete_single_selected_eta_pilot as frozen_eta_pilot
import run_stage4_ala_discrete_single_fixedalpha as coarse_single
import run_stage4_uniform_optimized as legacy


MEMBER_POLICY_VERSION = "stage4_ala_discrete_single_eta1p1_alpha20_followup_v1"
ETA = 1.1
ETA_TEXT = "1.1"
CANONICAL_ALPHA = 20.0
CANONICAL_RATIO = 0.8
CANONICAL_MAX_EVALUATED_STEPS = 10
CANONICAL_HISTORY = 7
CANONICAL_BUDGET = 254
CANONICAL_REPLICATE = "r1"
MESH_CV_MAX = 0.0065
MESH_RATIO_MAX = 1.03
ROUTING_OPTIONS = frozenset({"--output-dir", "--run-label"})
COMMON_RESULT_KEYS = frozen_eta_pilot.COMMON_RESULT_KEYS
DISCRETE_RESULT_KEYS = frozen_eta_pilot.DISCRETE_RESULT_KEYS
ETA_RESULT_KEYS = frozen_eta_pilot.ETA_RESULT_KEYS


def run_label() -> str:
    return (
        f"{MEMBER_POLICY_VERSION}_eta1p1_r08c10_a20_"
        f"N254_discrete_b040_{CANONICAL_REPLICATE}"
    )


def _replace_option(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"Expected one valued {flag} option")
    command[positions[0] + 1] = value


def canonical_reference_command(root: Path) -> list[str]:
    command, _ = coarse_single.single_run_command(
        root,
        CANONICAL_ALPHA,
        CANONICAL_BUDGET,
        CANONICAL_REPLICATE,
        "candidate",
    )
    return command


def assert_eta1p1_only_scientific_change(
    command: list[str], canonical: list[str], root: Path
) -> None:
    expected = list(canonical)
    expected[1] = str(
        (root / "src/run_ivm_v9_discrete_selected_eta_shim.py").resolve()
    )
    _replace_option(
        expected,
        "--output-dir",
        f"runs/{MEMBER_POLICY_VERSION}/members",
    )
    _replace_option(expected, "--run-label", run_label())
    expected.extend([eta_shim.ETA_FLAG, ETA_TEXT])
    if command != expected:
        raise RuntimeError(
            "Eta1.1 command is not the exact runner/routing/eta clone"
        )
    options = coarse_single.command_options(command)
    required = {
        "--alpha": "20",
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
        eta_shim.ETA_FLAG: ETA_TEXT,
    }
    for flag, expected_value in required.items():
        if options.get(flag) != expected_value:
            raise RuntimeError(f"Eta1.1 command changed frozen option {flag}")


def single_run_command(root: Path) -> tuple[list[str], str]:
    canonical = canonical_reference_command(root)
    command = list(canonical)
    command[1] = str(
        (root / "src/run_ivm_v9_discrete_selected_eta_shim.py").resolve()
    )
    _replace_option(
        command,
        "--output-dir",
        f"runs/{MEMBER_POLICY_VERSION}/members",
    )
    _replace_option(command, "--run-label", run_label())
    command.extend([eta_shim.ETA_FLAG, ETA_TEXT])
    assert_eta1p1_only_scientific_change(command, canonical, root)
    return command, run_label()


def compact_command_sha256(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def member_paths(root: Path) -> dict[str, Path | str]:
    label = run_label()
    return {
        "label": label,
        "result_dir": root / "runs" / MEMBER_POLICY_VERSION / "members",
        "audit": root / "runs" / MEMBER_POLICY_VERSION / "audits" / f"{label}.json",
        "member_log": root / "logs" / MEMBER_POLICY_VERSION / "members" / label,
        "launcher_log": root / "logs" / MEMBER_POLICY_VERSION / "launcher" / label,
    }


def _scalar(saved: np.lib.npyio.NpzFile, key: str):
    return np.asarray(saved[key]).item()


def _validate_result_iperp_only(result: Path) -> dict[str, object]:
    accessed = COMMON_RESULT_KEYS | DISCRETE_RESULT_KEYS | ETA_RESULT_KEYS
    if any("barrier" in key.lower() for key in accessed):
        raise RuntimeError("Eta1.1 result allowlist contains a barrier field")
    with np.load(result, allow_pickle=False) as saved:
        missing = sorted(accessed.difference(saved.files))
        if missing:
            raise RuntimeError(f"Result is missing: {', '.join(missing)}")
        if float(_scalar(saved, "alpha")) != CANONICAL_ALPHA or str(
            _scalar(saved, "alpha_mode")
        ) != "fixed":
            raise RuntimeError("Eta1.1 changed fixed alpha")
        active_alpha = np.asarray(saved["active_alpha"], dtype=float)
        if not active_alpha.size or not np.all(active_alpha == CANONICAL_ALPHA):
            raise RuntimeError("Eta1.1 changed alpha within the trajectory")
        required_scalars = {
            "frozen_subproblem": "discrete_b040",
            "discrete_stop_mode": "r08c10",
            "discrete_inner_max_steps": 10,
            "discrete_residual_ratio": 0.8,
            "discrete_history": 7,
            "reparam_mode": "state_curve",
            "state_curve_direct_objective_acceptance": "per_image",
            "stop_complete_iperp": -1.0,
        }
        for key, expected in required_scalars.items():
            if _scalar(saved, key) != expected:
                raise RuntimeError(f"Eta1.1 changed frozen result field {key}")
        if not bool(_scalar(saved, "discrete_all_returned_states_evaluated")):
            raise RuntimeError("Eta1.1 returned an unevaluated state")
        expected_id = eta_shim.shim_implementation_id(
            coarse_single.EXPECTED_IMPLEMENTATION_ID, ETA
        )
        identity = {
            "discrete_implementation_id": expected_id,
            "discrete_step_multiplier_policy_id": eta_shim.SHIM_POLICY_ID,
            "discrete_step_multiplier_base_implementation_id": (
                coarse_single.EXPECTED_IMPLEMENTATION_ID
            ),
            "discrete_step_multiplier_implementation_id": expected_id,
            "discrete_step_multiplier_eta": ETA,
            "discrete_step_multiplier_application_stage": eta_shim.APPLICATION_STAGE,
        }
        for key, expected in identity.items():
            if _scalar(saved, key) != expected:
                raise RuntimeError(f"Eta1.1 identity mismatch at {key}")
        if not bool(_scalar(saved, "discrete_step_multiplier_selector_isolated")):
            raise RuntimeError("Eta1.1 contaminated the selector")
        if not bool(
            _scalar(saved, "discrete_step_multiplier_dynamic_cap_reapplied")
        ):
            raise RuntimeError("Eta1.1 did not reapply the dynamic cap")
        if bool(_scalar(saved, "discrete_step_multiplier_energy_backtracking")):
            raise RuntimeError("Eta1.1 enabled forbidden energy backtracking")
        if int(
            _scalar(saved, "discrete_step_multiplier_extra_physical_evaluations")
        ) != 0:
            raise RuntimeError("Eta1.1 added physical evaluations")
        if int(
            _scalar(saved, "discrete_step_multiplier_aborted_selection_count")
        ) != 0:
            raise RuntimeError("Eta1.1 contains aborted selector events")
        if np.any(np.asarray(saved["subproblem_line_search_backtracks"], int) != 0):
            raise RuntimeError("Eta1.1 recorded forbidden backtracking")

        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        cv = np.asarray(saved["segment_length_cv"], dtype=float)
        ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
        if len({a.shape for a in (cost, i_perp, committed, cv, ratio)}) != 1 or (
            cost.ndim != 1 or cost.size != CANONICAL_BUDGET
        ):
            raise RuntimeError("Eta1.1 trajectory shapes are invalid")
        if np.any(np.diff(cost) <= 0) or not all(
            np.all(np.isfinite(a)) for a in (i_perp, cv, ratio)
        ):
            raise RuntimeError("Eta1.1 trajectory data are invalid")

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
        base = np.asarray(
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
        arrays = (
            eta_selection,
            requested,
            applied,
            cap_hit,
            dynamic,
            base,
            requested_max,
            applied_max,
        )
        if any(array.shape != selection.shape for array in arrays):
            raise RuntimeError("Eta1.1 audit arrays are misaligned")
        if not np.array_equal(selection, eta_selection) or not np.all(
            requested == ETA
        ):
            raise RuntimeError("Eta1.1 selector/request audit drifted")
        if not np.allclose(requested_max, ETA * base, rtol=1e-13, atol=1e-14):
            raise RuntimeError("Eta1.1 requested displacement mismatch")
        if np.any(applied <= 0.0) or np.any(applied > ETA * (1 + 1e-12)):
            raise RuntimeError("Eta1.1 applied multiplier is invalid")
        if np.any(applied_max > dynamic * (1 + 1e-12)) or not np.allclose(
            applied, applied_max / base, rtol=1e-12, atol=1e-14
        ):
            raise RuntimeError("Eta1.1 recap audit is invalid")
        if not np.array_equal(cap_hit, requested_max > dynamic):
            raise RuntimeError("Eta1.1 cap-hit audit is invalid")
        h0 = int(np.count_nonzero(selection == "H0"))
        lbfgs = int(np.count_nonzero(selection == "LBFGS"))
        if h0 != int(_scalar(saved, "discrete_h0_selection_count")) or lbfgs != int(
            _scalar(saved, "discrete_lbfgs_selection_count")
        ):
            raise RuntimeError("Eta1.1 direction counts are invalid")
        complete = committed.copy()
        complete[0] = True
        eligible = complete & (cv <= MESH_CV_MAX) & (ratio <= MESH_RATIO_MAX)
        return {
            "configuration_sha256": str(_scalar(saved, "configuration_sha256")),
            "initial_path_sha256": str(_scalar(saved, "initial_path_sha256")),
            "trajectory_record_count": int(cost.size),
            "final_optimization_NF": int(cost[-1]),
            "complete_state_NF": cost[complete].tolist(),
            "mesh_eligible_complete_state_NF": cost[eligible].tolist(),
            "mesh_eligible_complete_state_count": int(np.count_nonzero(eligible)),
            "h0_selection_count": h0,
            "lbfgs_selection_count": lbfgs,
            "eta_cap_hit_count": int(np.count_nonzero(cap_hit)),
            "eta_applied_multiplier_min": float(np.min(applied)),
            "eta_applied_multiplier_max": float(np.max(applied)),
            "result_fields_accessed": sorted(accessed),
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
    companion_audits = sorted(
        result_dir.glob(
            f"ivm_{label}_central_state_ivm_cfg*_budget254.audit.json"
        )
    )
    if existing or companion_audits or audit_path.exists() or log_dir.exists():
        target = (
            existing[0]
            if existing
            else companion_audits[0]
            if companion_audits
            else audit_path
        )
        raise FileExistsError(f"Refusing to overwrite eta1.1 member: {target}")
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
    summary = _validate_result_iperp_only(result)
    audit: dict[str, object] = {
        "member_policy_version": MEMBER_POLICY_VERSION,
        "member_role": "fixed_eta1p1_exploratory_followup",
        "eta": ETA,
        "alpha": CANONICAL_ALPHA,
        "fixed_alpha_throughout": True,
        "residual_ratio": CANONICAL_RATIO,
        "max_evaluated_steps": CANONICAL_MAX_EVALUATED_STEPS,
        "lbfgs_history": CANONICAL_HISTORY,
        "budget_per_image": CANONICAL_BUDGET,
        "replicate": CANONICAL_REPLICATE,
        "scientific_change_from_frozen_candidate": "only fixed post-selection eta=1.1",
        "scheduled_or_adaptive_eta": False,
        "selector_isolated_from_eta": True,
        "energy_backtracking": False,
        "extra_physical_evaluations_from_shim": 0,
        "stop_rule": "R/R0 <= 0.8 OR evaluated accepted steps == 10",
        "reparameterization_policy": "state_curve/per_image from first state",
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

