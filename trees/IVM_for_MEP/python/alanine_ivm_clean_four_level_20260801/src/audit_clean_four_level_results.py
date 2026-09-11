#!/usr/bin/env python3
"""Fail-closed structural, accounting, theory, and stability audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


REFERENCE_BARRIER_KCAL = 2.4560825131253083
COMMON_INITIAL_SHA256 = "cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 20.0, 30.0, 100.0, 1000.0, 10000.0, math.inf)
STAGES = {
    "stage1": "runs/stage1_exact_no_resample/*budget160.npz",
    "stage2": "runs/stage2_exact_resample/*budget160.npz",
    "stage3": "runs/stage3_inexact_baseline_resample/*budget220.npz",
    "stage4": "runs/stage4_inexact_optimized_resample/stage4_uniform_v3_a*_phaseA80_phaseB220.npz",
}


def label(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:g}".replace(".", "p").replace("-", "m")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(saved, key):
    return np.asarray(saved[key]).item()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def same_alpha(left: float, right: float) -> bool:
    return bool((math.isinf(left) and math.isinf(right)) or np.isclose(left, right))


def dense_path(root: Path, stage: str, rtol: float, alpha: float) -> Path:
    version = "_v3" if stage == "stage4" else ""
    return root / "runs/diagnostics" / (
        f"dense_barrier_clean_{stage}{version}_r{label(rtol)}_"
        f"a{label(alpha)}_common_checkpoints.npz"
    )


def stage_expected(stage: str, saved) -> tuple[float, dict]:
    if stage == "stage1":
        return float(scalar(saved, "plbfgs_kkt_rtol")), {
            "budget": 160,
            "inner_calls": 80,
            "reparam_every": 0,
            "reparam_mode": "geometric",
            "state_globalization": "proximal_only",
        }
    if stage == "stage2":
        return float(scalar(saved, "plbfgs_kkt_rtol")), {
            "budget": 160,
            "inner_calls": 80,
            "reparam_every": 1,
            "reparam_mode": "geometric",
            "state_globalization": "proximal_only",
        }
    if stage == "stage3":
        return 0.8, {
            "budget": 220,
            "inner_calls": 5,
            "plbfgs_kkt_rtol": 0.8,
            "reparam_every": 1,
            "reparam_mode": "geometric",
            "state_globalization": "proximal_only",
        }
    return 0.8, {}


def audit_phase(path: Path, alpha: float, phase: str) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        active_alpha = np.asarray(saved["active_alpha"], dtype=float)
        expected = {
            "method": "IVM",
            "alpha_mode": "fixed",
            "inner_calls": 15,
            "reparam_every": 1,
            "state_globalization": "iperp_monotone",
        }
        if phase == "A":
            expected.update(
                {
                    "plbfgs_kkt_rtol": 0.96,
                    "reparam_mode": "state_curve_projected",
                    "state_curve_direct_objective_acceptance": "path_sum",
                    "state_trust_fraction": 1.5,
                }
            )
        else:
            expected.update(
                {
                    "plbfgs_kkt_rtol": 0.8,
                    "reparam_mode": "state_curve",
                    "state_curve_direct_objective_acceptance": "per_image",
                    "state_trust_fraction": 2.0,
                }
            )
        checks = {}
        for key, value in expected.items():
            actual = scalar(saved, key)
            if isinstance(value, float):
                checks[key] = bool(np.isclose(float(actual), value))
            else:
                checks[key] = actual == value
        checks["active_alpha_fixed"] = bool(
            np.all(np.isclose(active_alpha, alpha, equal_nan=False))
            if not math.isinf(alpha)
            else np.all(np.isinf(active_alpha))
        )
        force = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        checks["force_counter_strict"] = bool(np.all(np.diff(force) > 0))
        checks["finite_raw_i_perp"] = bool(np.all(np.isfinite(saved["i_perp"])))
        return {
            "path": str(path),
            "sha256": file_sha256(path),
            "checks": checks,
            "passed": all(checks.values()),
        }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    records = []
    failures = []
    stage_counts = {}
    stage_alpha_rtol = {}
    for stage, pattern in STAGES.items():
        if stage == "stage4":
            # The directory also retains alpha=12/15/18 tuning pilots.  This
            # structural audit is defined only on the frozen ten-alpha grid.
            paths = [
                root
                / "runs/stage4_inexact_optimized_resample"
                / f"stage4_uniform_v3_a{label(alpha)}_phaseA80_phaseB220.npz"
                for alpha in ALPHAS
            ]
        else:
            paths = sorted(root.glob(pattern))
        stage_counts[stage] = len(paths)
        stage_alpha_rtol[stage] = []
        expected_count = 20 if stage in {"stage1", "stage2"} else 10
        if len(paths) != expected_count:
            failures.append(f"{stage}: expected {expected_count} results, found {len(paths)}")
        for path in paths:
            with np.load(path, allow_pickle=False) as saved:
                alpha = float(scalar(saved, "alpha"))
                rtol, expected = stage_expected(stage, saved)
                initial_hash_key = (
                    "common_initial_path_sha256" if stage == "stage4" else "initial_path_sha256"
                )
                checks = {
                    "common_initial_path": str(scalar(saved, initial_hash_key))
                    == COMMON_INITIAL_SHA256,
                    "alpha_on_grid": any(same_alpha(alpha, item) for item in ALPHAS),
                }
                if stage == "stage4":
                    force = np.asarray(saved["total_force_evaluations"], dtype=int)
                    checks.update(
                        {
                            "policy_version": str(scalar(saved, "policy_version"))
                            == "stage4_uniform_v3",
                            "fixed_across_both_phases": str(scalar(saved, "alpha_mode"))
                            == "fixed_across_both_phases",
                            "online_stop_target_uniform": np.isclose(
                                float(scalar(saved, "stop_complete_iperp")), 1.5e-4
                            ),
                            "force_counter_strict": bool(np.all(np.diff(force) > 0)),
                            "raw_i_perp_monotone": bool(
                                np.count_nonzero(np.diff(saved["i_perp"]) > 1e-12) == 0
                            ),
                        }
                    )
                    phase_a = audit_phase(Path(str(scalar(saved, "phase_a_result"))), alpha, "A")
                    phase_b = audit_phase(Path(str(scalar(saved, "phase_b_result"))), alpha, "B")
                    checks["phase_a_theory_configuration"] = phase_a["passed"]
                    checks["phase_b_theory_configuration"] = phase_b["passed"]
                    provenance = {
                        "materialization_mode": str(scalar(saved, "materialization_mode")),
                        "independent_online_rerun_status": str(
                            scalar(saved, "independent_online_rerun_status")
                        ),
                    }
                    if np.isclose(alpha, 20.0):
                        checks.update(
                            {
                                "best_alpha_independent_online": provenance["materialization_mode"]
                                == "independent_online_execution",
                                "best_alpha_online_confirmed": provenance[
                                    "independent_online_rerun_status"
                                ]
                                == "confirmed",
                                "best_alpha_stop_fired": bool(scalar(saved, "stop_rule_fired")),
                                "best_alpha_terminal_commit": bool(
                                    scalar(saved, "terminal_reparam_committed")
                                ),
                            }
                        )
                else:
                    force = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
                    active_alpha = np.asarray(saved["active_alpha"], dtype=float)
                    checks.update(
                        {
                            "method_ivm": str(scalar(saved, "method")) == "IVM",
                            "alpha_fixed": str(scalar(saved, "alpha_mode")) == "fixed",
                            "active_alpha_fixed": bool(
                                np.all(np.isclose(active_alpha, alpha))
                                if not math.isinf(alpha)
                                else np.all(np.isinf(active_alpha))
                            ),
                            "force_counter_strict": bool(np.all(np.diff(force) > 0)),
                            "finite_raw_i_perp": bool(np.all(np.isfinite(saved["i_perp"]))),
                        }
                    )
                    for key, value in expected.items():
                        actual = scalar(saved, key)
                        checks[f"config_{key}"] = (
                            bool(np.isclose(float(actual), value))
                            if isinstance(value, (int, float))
                            else actual == value
                        )
                    phase_a = phase_b = None
                    provenance = None
                diagnostic = dense_path(root, stage, rtol, alpha)
                checks["dense_diagnostic_exists"] = diagnostic.exists()
                if diagnostic.exists():
                    with np.load(diagnostic, allow_pickle=False) as dense:
                        checks["dense_initial_hash"] = str(
                            scalar(dense, "source_initial_path_sha256")
                        ) == COMMON_INITIAL_SHA256
                        checks["dense_calls_strict"] = bool(
                            np.all(np.diff(dense["calls"]) > 0)
                        )
                        checks["dense_barrier_finite"] = bool(
                            np.all(np.isfinite(dense["adaptive_barrier_kcal"]))
                        )
                stage_alpha_rtol[stage].append(
                    {"alpha": "inf" if math.isinf(alpha) else alpha, "rtol": rtol}
                )
                record = {
                    "stage": stage,
                    "alpha": "inf" if math.isinf(alpha) else alpha,
                    "rtol": rtol,
                    "result": str(path),
                    "sha256": file_sha256(path),
                    "checks": checks,
                    "passed": all(checks.values()),
                    "dense_diagnostic": str(diagnostic),
                    "phase_a": phase_a,
                    "phase_b": phase_b,
                    "provenance": provenance,
                }
                if not record["passed"]:
                    failures.append(
                        f"{stage} alpha={record['alpha']}: "
                        + ", ".join(key for key, value in checks.items() if not value)
                    )
                records.append(record)

    for stage in ("stage1", "stage2"):
        for rtol in (0.1, 0.2):
            got = [item["alpha"] for item in stage_alpha_rtol[stage] if np.isclose(item["rtol"], rtol)]
            if len(got) != len(ALPHAS):
                failures.append(f"{stage} rtol={rtol}: incomplete alpha grid")
    if len(stage_alpha_rtol["stage3"]) != len(ALPHAS):
        failures.append("stage3: incomplete alpha grid")

    dedicated = (
        root
        / "runs/diagnostics/dense_barrier_clean_stage4_v3_a20_complete_macrostates.npz"
    )
    with np.load(dedicated, allow_pickle=False) as dense:
        dense_calls = np.asarray(dense["calls"], dtype=int)
        dense_barrier = np.asarray(dense["adaptive_barrier_kcal"], dtype=float)
    dense_error = np.abs(dense_barrier - REFERENCE_BARRIER_KCAL)
    first_below = np.flatnonzero(dense_error <= 0.05)
    barrier_stability = {
        "source": str(dedicated),
        "calls": dense_calls.tolist(),
        "barrier_kcal": dense_barrier.tolist(),
        "absolute_error_kcal": dense_error.tolist(),
        "barrier_rises": int(np.count_nonzero(np.diff(dense_barrier) > 1e-12)),
        "absolute_error_rises": int(np.count_nonzero(np.diff(dense_error) > 1e-12)),
        "maximum_error_after_first_below_0p05": float(
            np.max(dense_error[first_below[0] :]) if first_below.size else math.inf
        ),
        "terminal_error_kcal": float(dense_error[-1]),
    }
    barrier_stability["strict_no_oscillation_passed"] = bool(
        barrier_stability["absolute_error_rises"] == 0
    )

    online_verification_path = root / "reports/stage4_v3_online_prefix_verification.json"
    online_verification = json.loads(online_verification_path.read_text())
    all_methods_path = (
        root
        / "figures/clean_four_level_audit/all_methods_stage4_v4_pareto/"
        "alanine_stage4_v4_pareto_all_methods_audit.json"
    )
    all_methods = json.loads(all_methods_path.read_text())
    structural_passed = not failures and online_verification["status"] == "passed"
    strict_acceptance_passed = bool(
        structural_passed
        and barrier_stability["strict_no_oscillation_passed"]
        and all_methods["ivm_beats_ode_i_perp"]
        and all_methods["ivm_beats_ode_barrier_error"]
    )
    report = {
        "protocol": {
            "common_initial_path_sha256": COMMON_INITIAL_SHA256,
            "alpha_grid": ["inf" if math.isinf(value) else value for value in ALPHAS],
            "stage1_exact_like_definition": "finite KKT-controlled solve, r=0.2/0.1, inner cap 80, no resampling",
            "stage2_exact_like_definition": "same operational exact-like solve plus geometric resampling",
            "stage3_definition": "baseline inexact, r=0.8, five inner calls, geometric resampling",
            "stage4_definition": "uniform two-phase fixed-alpha state-IVM with independently audited online stop",
            "optimization_axis": "whole-path force evaluations only; dense diagnostics excluded",
        },
        "stage_counts": stage_counts,
        "structural_theory_accounting_passed": structural_passed,
        "failures": failures,
        "records": records,
        "online_prefix_verification": online_verification,
        "alpha20_dense_barrier_stability": barrier_stability,
        "all_methods_audit": all_methods,
        "strict_user_acceptance_passed": strict_acceptance_passed,
        "status": (
            "passed"
            if strict_acceptance_passed
            else "structurally_passed_but_strict_barrier_stability_unmet"
            if structural_passed
            else "failed"
        ),
    }
    output = root / "reports/clean_four_level_fail_closed_audit.json"
    output.write_text(json.dumps(report, indent=2, default=json_default) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "stage_counts": stage_counts,
                "structural_theory_accounting_passed": structural_passed,
                "strict_user_acceptance_passed": strict_acceptance_passed,
                "status": report["status"],
                "failures": failures,
                "alpha20_dense_barrier_stability": barrier_stability,
            },
            indent=2,
            default=json_default,
        )
    )
    if not structural_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
