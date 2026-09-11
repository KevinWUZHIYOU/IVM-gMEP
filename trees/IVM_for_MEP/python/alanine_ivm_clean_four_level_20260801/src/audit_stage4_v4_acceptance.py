#!/usr/bin/env python3
"""Fail-closed acceptance audit for the clean four-level v4 Ala result."""

from __future__ import annotations

import glob
import hashlib
import json
import math
from pathlib import Path

import numpy as np


REFERENCE_BARRIER_KCAL = 2.4560825131253083
ODE_TERMINAL_IPERP = 2.0406804672111045e-4
ODE_TERMINAL_BARRIER_ERROR = 0.03607305433804209
ODE_IPERP_FIRST_PASSAGE_NF = 83522
EXPECTED_ALPHAS = (0.01, 0.1, 1.0, 10.0, 20.0, 30.0, 100.0, 1000.0, 10000.0, math.inf)


def scalar(saved, key):
    return np.asarray(saved[key]).item()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def alpha_tag(alpha: float) -> str:
    return "inf" if math.isinf(alpha) else f"{alpha:g}".replace(".", "p")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    checks = {}
    evidence = {}

    stage1 = sorted(
        p for p in (root / "runs/stage1_exact_no_resample").glob("*.npz")
        if "budget160" in p.name
    )
    stage2 = sorted(
        p for p in (root / "runs/stage2_exact_resample").glob("*.npz")
        if "budget160" in p.name
    )
    stage3 = sorted(
        (root / "runs/stage3_inexact_baseline_resample").glob("*budget220.npz")
    )
    stage4 = [
        root / "runs/stage4_inexact_optimized_resample"
        / f"stage4_uniform_v3_a{alpha_tag(alpha)}_phaseA34_phaseB220.npz"
        for alpha in EXPECTED_ALPHAS
    ]
    archives = [
        root / "runs/pareto_archive"
        / f"pareto_clean_stage4_v4_a{alpha_tag(alpha)}_phaseA34_phaseB220.npz"
        for alpha in EXPECTED_ALPHAS
    ]
    checks["stage1_exact_like_no_resample_has_20_main_results"] = len(stage1) == 20
    checks["stage2_exact_like_resample_has_20_main_results"] = len(stage2) == 20
    checks["stage3_baseline_inexact_resample_has_10_main_results"] = len(stage3) == 10
    checks["stage4_uniform_v4_has_10_alpha_results"] = all(p.exists() for p in stage4)
    checks["stage4_uniform_v4_has_10_pareto_archives"] = all(p.exists() for p in archives)
    evidence["four_level_counts"] = {
        "stage1": len(stage1), "stage2": len(stage2), "stage3": len(stage3),
        "stage4": sum(p.exists() for p in stage4),
        "stage4_archives": sum(p.exists() for p in archives),
    }

    common_hashes = set()
    archive_status = {}
    accounting_ok = True
    reference_free = True
    source_alpha_ok = True
    fixed_alpha_ok = True
    for alpha, combined_path, archive_path in zip(EXPECTED_ALPHAS, stage4, archives):
        if not combined_path.exists() or not archive_path.exists():
            continue
        with np.load(combined_path, allow_pickle=False) as combined, np.load(
            archive_path, allow_pickle=False
        ) as archive:
            common_hashes.add(str(scalar(combined, "common_initial_path_sha256")))
            source_alpha_ok &= bool(
                (math.isinf(alpha) and math.isinf(float(scalar(combined, "alpha"))))
                or np.isclose(float(scalar(combined, "alpha")), alpha)
            )
            fixed_alpha_ok &= bool(
                str(scalar(combined, "alpha_mode")) == "fixed_across_both_phases"
            )
            reference_free &= not bool(scalar(archive, "reference_barrier_used"))
            candidate_cost = np.asarray(
                archive["candidate_screen_force_equivalent_cost"], dtype=int
            )
            accounting_ok &= bool(
                int(np.sum(candidate_cost))
                == int(scalar(archive, "screen_force_equivalent_evaluations"))
            )
            archive_status["inf" if math.isinf(alpha) else f"{alpha:g}"] = bool(
                scalar(archive, "target_reached")
            )
    expected_common_hash = "cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e"
    checks["all_stage4_runs_share_clean_initial_path"] = common_hashes == {expected_common_hash}
    checks["stage4_alpha_values_match_grid"] = source_alpha_ok
    checks["alpha_is_fixed_across_both_phases"] = fixed_alpha_ok
    checks["pareto_screen_is_reference_free"] = reference_free
    checks["screen_force_accounting_is_exact"] = accounting_ok
    checks["alpha20_is_unique_target_reaching_run"] = (
        archive_status.get("20") is True
        and sum(bool(value) for value in archive_status.values()) == 1
    )
    evidence["target_reached_by_alpha"] = archive_status
    evidence["common_initial_path_sha256"] = sorted(common_hashes)

    combined_path = stage4[EXPECTED_ALPHAS.index(20.0)]
    archive_path = archives[EXPECTED_ALPHAS.index(20.0)]
    dense_path = (
        root / "runs/diagnostics"
        / "dense_barrier_clean_stage4_pareto_screen_a20_phaseA34_phaseB220.npz"
    )
    with np.load(combined_path, allow_pickle=False) as combined, np.load(
        archive_path, allow_pickle=False
    ) as archive, np.load(dense_path, allow_pickle=False) as dense:
        selected_call = int(scalar(archive, "selected_call"))
        selected_i = float(scalar(archive, "selected_i_perp"))
        total_charged = int(
            scalar(archive, "total_charged_force_equivalent_evaluations")
        )
        selected_path = np.asarray(archive["selected_path"], dtype=float)
        labels = np.asarray(combined["stored_force_evaluations"], dtype=int)
        path_matches = np.flatnonzero(labels == selected_call)
        source_path = np.asarray(combined["stored_paths"][int(path_matches[-1])], dtype=float)
        selected_path_exact = np.array_equal(selected_path, source_path)
        phase_b = np.load(str(scalar(combined, "phase_b_result")), allow_pickle=False)
        offset = int(scalar(combined, "phase_transition_force_offset"))
        committed_force = offset + np.asarray(
            phase_b["total_internal_joint_evaluations"], dtype=int
        )[np.asarray(phase_b["reparam_committed"], dtype=bool)]
        selected_is_complete = selected_call in set(int(x) for x in committed_force)
        active_alpha_a = np.asarray(
            np.load(str(scalar(combined, "phase_a_result")), allow_pickle=False)["active_alpha"],
            dtype=float,
        )
        active_alpha_b = np.asarray(phase_b["active_alpha"], dtype=float)
        active_alpha_exact = bool(
            np.all(active_alpha_a == 20.0) and np.all(active_alpha_b == 20.0)
        )
        phase_b.close()

        accepted_calls = np.asarray(archive["accepted_calls"], dtype=int)
        accepted_indices = [
            int(np.flatnonzero(np.asarray(dense["calls"], dtype=int) == call)[-1])
            for call in accepted_calls
        ]
        accepted_barrier20 = np.asarray(dense["barrier_20_kcal"], dtype=float)[
            accepted_indices
        ]
        accepted_error = np.abs(accepted_barrier20 - REFERENCE_BARRIER_KCAL)
        accepted_i = np.asarray(archive["candidate_i_perp"], dtype=float)[
            np.asarray(archive["candidate_accepted"], dtype=bool)
        ]
        selected_barrier = float(accepted_barrier20[-1])
        selected_error = float(accepted_error[-1])
        raw_calls = np.asarray(dense["calls"], dtype=int)
        raw_prefix = raw_calls <= selected_call
        raw_error = np.abs(
            np.asarray(dense["barrier_20_kcal"], dtype=float)[raw_prefix]
            - REFERENCE_BARRIER_KCAL
        )
        screen_cost = int(scalar(archive, "screen_force_equivalent_evaluations"))

    checks["selected_path_is_exact_stored_state"] = selected_path_exact
    checks["selected_state_is_completed_reparameterized_macrostate"] = selected_is_complete
    checks["active_alpha_is_exact_integer20_in_both_phases"] = active_alpha_exact
    checks["published_i_perp_has_zero_rises"] = not np.any(np.diff(accepted_i) > 1e-12)
    checks["published_dense_barrier_error_has_zero_rises"] = not np.any(
        np.diff(accepted_error) > 1e-8
    )
    checks["selected_i_perp_beats_ode_neb"] = selected_i < ODE_TERMINAL_IPERP
    checks["selected_dense_barrier_error_beats_ode_neb"] = (
        selected_error < ODE_TERMINAL_BARRIER_ERROR
    )
    speedup = ODE_IPERP_FIRST_PASSAGE_NF / total_charged
    checks["charged_cost_speedup_is_at_least_10x"] = speedup >= 10.0
    checks["raw_rejected_candidate_shoulder_is_disclosed"] = bool(
        np.count_nonzero(np.diff(raw_error) > 1e-8) > 0
        and float(np.max(raw_error)) > float(np.max(accepted_error))
    )
    evidence["selected_alpha20"] = {
        "optimization_force_evaluations": selected_call,
        "screen_force_equivalent_evaluations": screen_cost,
        "total_charged_force_equivalent_evaluations": total_charged,
        "i_perp": selected_i,
        "dense_barrier20_kcal_per_mol": selected_barrier,
        "dense_barrier20_abs_error_kcal_per_mol": selected_error,
        "ode_terminal_i_perp": ODE_TERMINAL_IPERP,
        "ode_terminal_barrier_abs_error_kcal_per_mol": ODE_TERMINAL_BARRIER_ERROR,
        "ode_first_passage_force_evaluations": ODE_IPERP_FIRST_PASSAGE_NF,
        "charged_cost_speedup": speedup,
        "accepted_calls": accepted_calls.tolist(),
        "accepted_i_perp": accepted_i.tolist(),
        "accepted_barrier20_kcal_per_mol": accepted_barrier20.tolist(),
        "accepted_barrier20_abs_error_kcal_per_mol": accepted_error.tolist(),
        "accepted_i_perp_rises": int(np.count_nonzero(np.diff(accepted_i) > 1e-12)),
        "accepted_barrier_error_rises": int(
            np.count_nonzero(np.diff(accepted_error) > 1e-8)
        ),
        "raw_candidate_barrier_error_max": float(np.max(raw_error)),
        "raw_candidate_barrier_error_rises": int(
            np.count_nonzero(np.diff(raw_error) > 1e-8)
        ),
        "selected_path_sha256": array_sha256(selected_path),
    }

    path_dir = root / "figures/clean_four_level_audit/all_methods_stage4_v4_pareto"
    descent_dir = (
        root
        / "figures/clean_four_level_audit/"
        "all_methods_stage4_v4_pareto_linear_speedup_v5"
    )
    alpha_dir = (
        root
        / "figures/clean_four_level_audit/alpha_common5000_decades_v10_titles"
    )
    stems = (
        path_dir / "alanine_integer20_all_methods_1x4_path_evolution",
        descent_dir
        / "alanine_integer20_all_methods_two_metric_descent_linear_cost_speedup",
        alpha_dir
        / "alanine_v10_cut5000_decades_protocol_titles_"
        "alpha_budget_aware_cost_to_precision_stage1_vs_stage4",
    )
    expected_plots = [stem.with_suffix(suffix) for stem in stems for suffix in (".pdf", ".svg", ".png")]
    checks["all_current_formal_figures_exist_in_pdf_svg_png"] = all(
        p.exists() for p in expected_plots
    )
    evidence["final_figures"] = [str(path) for path in expected_plots]

    overall = all(checks.values())
    report = {
        "status": "accepted" if overall else "rejected",
        "overall_acceptance": overall,
        "acceptance_scope": (
            "published complete-state Pareto incumbent trace; raw trial/complete "
            "candidates remain separately disclosed"
        ),
        "checks": checks,
        "evidence": evidence,
    }
    output = root / "reports/stage4_v4_fail_closed_acceptance.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
