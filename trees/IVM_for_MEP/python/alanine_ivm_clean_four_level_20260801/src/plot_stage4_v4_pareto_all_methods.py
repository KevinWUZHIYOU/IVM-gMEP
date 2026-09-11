#!/usr/bin/env python3
"""Use the established Ala plotting code with the clean v4 Pareto IVM archive."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def load_legacy_module(repo: Path):
    legacy_src = repo / "python/alanine_v10_chignolin_v9_fixed_alpha_20260730/src"
    sys.path.insert(0, str(legacy_src))
    source = legacy_src / "plot_ala_integer20_all_methods.py"
    specification = importlib.util.spec_from_file_location(
        "legacy_ala_all_methods_v4", source
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def scalar(saved, key):
    return np.asarray(saved[key]).item()


def path_at_force(saved, force: int) -> np.ndarray:
    labels = np.asarray(saved["stored_force_evaluations"], dtype=int)
    matches = np.flatnonzero(labels == force)
    if not matches.size:
        raise RuntimeError(f"Stored path is missing at NF={force}")
    return np.asarray(saved["stored_paths"][int(matches[-1])], dtype=float)


def load_pareto_ivm(
    module,
    combined_path: Path,
    archive_path: Path,
    dense_prefix_path: Path,
    dense_screen_path: Path,
    initial_path: np.ndarray,
):
    combined = np.load(combined_path, allow_pickle=False)
    archive = np.load(archive_path, allow_pickle=False)
    dense_prefix = np.load(dense_prefix_path, allow_pickle=False)
    dense_screen = np.load(dense_screen_path, allow_pickle=False)
    if not bool(scalar(archive, "target_reached")):
        raise RuntimeError("The selected Pareto archive did not reach its target")
    if not np.isclose(float(scalar(archive, "alpha")), 20.0):
        raise RuntimeError("The selected Pareto archive is not integer alpha=20")
    if bool(scalar(archive, "reference_barrier_used")):
        raise RuntimeError("Pareto screen must be reference-free")

    candidate_calls = np.asarray(archive["candidate_calls"], dtype=int)
    candidate_i = np.asarray(archive["candidate_i_perp"], dtype=float)
    candidate_cost = np.asarray(
        archive["candidate_screen_force_equivalent_cost"], dtype=int
    )
    candidate_accepted = np.asarray(archive["candidate_accepted"], dtype=bool)
    cumulative_screen = np.cumsum(candidate_cost)
    accepted_calls = candidate_calls[candidate_accepted]
    accepted_cost = accepted_calls + cumulative_screen[candidate_accepted]
    accepted_i = candidate_i[candidate_accepted]
    selected_call = int(scalar(archive, "selected_call"))
    selected_force = int(
        scalar(archive, "total_charged_force_equivalent_evaluations")
    )
    if int(accepted_calls[-1]) != selected_call or int(accepted_cost[-1]) != selected_force:
        raise RuntimeError("Selected Pareto cost is inconsistent with the accepted trace")

    raw_force = np.asarray(combined["total_force_evaluations"], dtype=int)
    raw_i = np.asarray(combined["i_perp"], dtype=float)
    prefix_mask = raw_force < int(candidate_calls[0])
    force = np.concatenate([raw_force[prefix_mask], accepted_cost])
    iperp = np.concatenate([raw_i[prefix_mask], accepted_i])
    if np.count_nonzero(np.diff(iperp) > 1e-12):
        raise RuntimeError("Published Pareto I_perp trace is not monotone")

    screen_calls = np.asarray(dense_screen["calls"], dtype=int)
    barrier20_by_call = {
        int(call): float(value)
        for call, value in zip(screen_calls, dense_screen["barrier_20_kcal"])
    }
    accepted_barrier = np.asarray(
        [barrier20_by_call[int(call)] for call in accepted_calls], dtype=float
    )
    prefix_barrier_calls = np.asarray(dense_prefix["calls"], dtype=int)
    prefix_barrier_mask = prefix_barrier_calls < int(candidate_calls[0])
    barrier_force = np.concatenate(
        [prefix_barrier_calls[prefix_barrier_mask], accepted_cost]
    )
    barrier = np.concatenate(
        [
            np.asarray(dense_prefix["barrier_20_kcal"], dtype=float)[
                prefix_barrier_mask
            ],
            accepted_barrier,
        ]
    )

    prefix_path_calls = [20, 60, 140, 300, 440]
    path_forces = list(prefix_path_calls)
    paths = [path_at_force(combined, call) for call in prefix_path_calls]
    for call, charged in zip(accepted_calls, accepted_cost):
        path_forces.append(int(charged))
        paths.append(path_at_force(combined, int(call)))

    raw_candidates = screen_calls <= selected_call
    raw_barrier_error = np.abs(
        np.asarray(dense_screen["barrier_20_kcal"], dtype=float)[raw_candidates]
        - module.REFERENCE_BARRIER_KCAL
    )
    accepted_error = np.abs(accepted_barrier - module.REFERENCE_BARRIER_KCAL)
    disclosure = {
        "raw_complete_candidate_barrier_error_max": float(np.max(raw_barrier_error)),
        "raw_complete_candidate_barrier_error_rises": int(
            np.count_nonzero(np.diff(raw_barrier_error) > 1e-8)
        ),
        "published_incumbent_barrier_error_rises": int(
            np.count_nonzero(np.diff(accepted_error) > 1e-8)
        ),
        "accepted_calls": accepted_calls.tolist(),
        "accepted_charged_force": accepted_cost.tolist(),
        "accepted_i_perp": accepted_i.tolist(),
        "accepted_barrier20_kcal": accepted_barrier.tolist(),
        "accepted_barrier20_abs_error": accepted_error.tolist(),
        "screen_force_equivalent_evaluations": int(
            scalar(archive, "screen_force_equivalent_evaluations")
        ),
    }
    return module.MethodSeries(
        key="ivm",
        label="IVM ($\\alpha=20$, Pareto archive)",
        short_label="IVM",
        color="#0072B2",
        marker="^",
        linestyle="-",
        force=force,
        iperp=iperp,
        barrier_force=barrier_force,
        barrier=barrier,
        selected_call=selected_call,
        selected_force=selected_force,
        selected_iperp=float(scalar(archive, "selected_i_perp")),
        selected_barrier=float(accepted_barrier[-1]),
        selected_path=np.asarray(archive["selected_path"], dtype=float),
        initial_path=np.asarray(initial_path, dtype=float),
        path_force=np.asarray(path_forces, dtype=int),
        paths=np.asarray(paths, dtype=float),
        configuration=(
            "clean-start stage-4 v4 Pareto incumbent; fixed alpha=20; "
            "phase A=34; phase B<=220; barrier10 reference-free screen; "
            "all screen OpenMM calls charged"
        ),
        result_path=archive_path,
        dense_path=dense_screen_path,
    ), disclosure


def main() -> None:
    study = Path(__file__).resolve().parents[1]
    repo = Path(__file__).resolve().parents[3]
    legacy = repo / "python/alanine_v10_chignolin_v9_fixed_alpha_20260730"
    module = load_legacy_module(repo)
    output_dir = study / "figures/clean_four_level_audit/all_methods_stage4_v4_pareto"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "combined": study / "runs/stage4_inexact_optimized_resample/stage4_uniform_v3_a20_phaseA34_phaseB220.npz",
        "archive": study / "runs/pareto_archive/pareto_clean_stage4_v4_a20_phaseA34_phaseB220.npz",
        "dense_prefix": study / "runs/diagnostics/dense_barrier_clean_stage4_v4_a20_phaseA34_preactivation.npz",
        "dense_screen": study / "runs/diagnostics/dense_barrier_clean_stage4_pareto_screen_a20_phaseA34_phaseB220.npz",
        "original": legacy / "runs/competitor_screen_original_finalists/competitor_screen_original_finalists_finalists20260731_original_nolambda_dt0p014_r1_cfg587b411dd523_budget3000.npz",
        "dense_original": legacy / "runs/diagnostics/dense_barrier_ala_original_dt0p014_finalist.npz",
        "static": legacy / "runs/neb/neb_finalists20260731_central_k30_alpha0p013_static_budget3000.npz",
        "dense_static": legacy / "runs/diagnostics/dense_barrier_ala_static_k30_dt0p013_finalist.npz",
        "ode": legacy / "formal_figure_inputs/ala_0to1_N22_k50_alpha0.0001_ode_results.npz",
        "ode_metrics": legacy / "formal_figure_inputs/metrics_ala_0to1_N22_k50_alpha0.0001_ode_results.npz",
        "dense_ode": legacy / "runs/diagnostics/dense_barrier_ala_ode_neb_k50_logsample_dense.npz",
        "bfgs": legacy / "runs/competitor_screen_bfgs_diagnostic_finalists/competitor_screen_bfgs_diagnostic_finalists_diagnostic_finalists20260731_bfgs_lam0p3_inner1_r1_nocap_cfg3958c11e3826_budget300.npz",
        "dense_bfgs": legacy / "runs/diagnostics/dense_barrier_ala_obsm_bfgs_lam0p3_r1_finalist.npz",
        "fire": legacy / "runs/competitor_screen_fire_finalists/competitor_screen_fire_finalists_finalists20260731_fire_lam0p85_inner3_r1_nocap_cfg3416aa96d5bf_budget2000.npz",
        "dense_fire": legacy / "runs/diagnostics/dense_barrier_ala_obsm_fire_lam0p85_inner3_dt0p1_finalist.npz",
        "common_initial": study / "inputs/common_initial_path_equidistant.npz",
        "rama": legacy / "formal_figure_inputs/ala2_scan_cache.pkl",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    common_initial = np.asarray(
        np.load(paths["common_initial"])["initial_path"], dtype=float
    )
    ivm, disclosure = load_pareto_ivm(
        module,
        paths["combined"],
        paths["archive"],
        paths["dense_prefix"],
        paths["dense_screen"],
        common_initial,
    )
    original = module.load_generic(
        key="original_string", label="Original String ($dt=0.014$)", short_label="Original String",
        color="#009E73", marker="D", linestyle=(0, (1.5, 1.4)), result_path=paths["original"],
        dense_path=paths["dense_original"], selected_call=3000,
        configuration="Original String; dt=0.014; reparameterize every step"
    )
    static = module.load_generic(
        key="static_neb", label="Static NEB ($k=30$, $dt=0.013$)", short_label="Static NEB",
        color="#CC79A7", marker="X", linestyle=(0, (3, 1.4, 1, 1.4)), result_path=paths["static"],
        dense_path=paths["dense_static"], selected_call=3000,
        configuration="static central-tangent NEB; k=30; dt=0.013"
    )
    ode = module.load_ode(
        paths["ode"], paths["ode_metrics"], paths["dense_ode"], common_initial
    )
    bfgs = module.load_generic(
        key="obsm_bfgs", label="OBSM--BFGS (bonded $H_0$)", short_label="OBSM--BFGS",
        color="#E69F00", marker="v", linestyle=(0, (6, 1.5, 1.2, 1.5)), result_path=paths["bfgs"],
        dense_path=paths["dense_bfgs"], selected_call=91,
        configuration="OBSM full dense BFGS; lambda=0.3; one inner step; bonded H0",
        truncate_paths_at_selection=True
    )
    fire = module.load_generic(
        key="obsm_fire", label="OBSM--FIRE", short_label="OBSM--FIRE", color="#56B4E9",
        marker="P", linestyle=(0, (2.5, 1.4)), result_path=paths["fire"],
        dense_path=paths["dense_fire"], selected_call=1999,
        configuration="OBSM--FIRE; lambda=0.85; 3 inner steps; dt=0.1"
    )
    methods = [ivm, ode, original, static, bfgs, fire]
    initial_differences = {
        method.key: float(np.max(np.abs(method.initial_path - common_initial)))
        for method in methods
    }
    if any(value > 5e-7 for value in initial_differences.values()):
        raise RuntimeError(f"Initial path mismatch: {initial_differences}")
    with paths["rama"].open("rb") as handle:
        rama = module._ContourCacheUnpickler(handle).load()
    module.configure_original_style()
    descent = module.plot_descent(methods, output_dir)
    cost_outputs, cost_rows, speedups = module.plot_cost_and_paths(
        methods, rama, output_dir
    )
    # The manuscript path-evolution figure intentionally compares only the
    # selected fixed-alpha IVM path with the strongest NEB trajectory.  The
    # full six-method quantitative comparison remains in the descent figure.
    evolution_methods = [ivm, ode]
    evolution = module.plot_path_evolution(evolution_methods, rama, output_dir)
    summary_path = output_dir / "alanine_stage4_v4_pareto_selected_methods.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "method", "selected_NF", "selected_i_perp",
            "selected_dense_barrier", "barrier_abs_error"
        ])
        for method in methods:
            writer.writerow([
                method.short_label, method.selected_force, method.selected_iperp,
                method.selected_barrier,
                abs(method.selected_barrier - module.REFERENCE_BARRIER_KCAL),
            ])
    audit = {
        "ivm_policy": "stage4_uniform_v4_pareto_archive",
        "ivm_alpha": 20,
        "ivm_selected_optimization_NF": 2507,
        "ivm_screen_force_equivalent_NF": disclosure[
            "screen_force_equivalent_evaluations"
        ],
        "ivm_selected_total_charged_NF": ivm.selected_force,
        "ivm_selected_i_perp": ivm.selected_iperp,
        "ivm_selected_dense_barrier": ivm.selected_barrier,
        "ivm_barrier_abs_error": abs(
            ivm.selected_barrier - module.REFERENCE_BARRIER_KCAL
        ),
        "ode_terminal_i_perp": ode.selected_iperp,
        "ode_terminal_barrier_abs_error": abs(
            ode.selected_barrier - module.REFERENCE_BARRIER_KCAL
        ),
        "ivm_beats_ode_i_perp": bool(ivm.selected_iperp < ode.selected_iperp),
        "ivm_beats_ode_barrier_error": bool(
            abs(ivm.selected_barrier - module.REFERENCE_BARRIER_KCAL)
            < abs(ode.selected_barrier - module.REFERENCE_BARRIER_KCAL)
        ),
        "pareto_disclosure": disclosure,
        "first_passage_speedups": speedups,
        "initial_path_max_abs_differences": initial_differences,
        "outputs": {
            "descent": descent,
            "cost_and_paths": cost_outputs,
            "evolution": evolution,
        },
        "path_evolution_methods": [
            method.short_label for method in evolution_methods
        ],
        "cost_rows": cost_rows,
    }
    (output_dir / "alanine_stage4_v4_pareto_all_methods_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
