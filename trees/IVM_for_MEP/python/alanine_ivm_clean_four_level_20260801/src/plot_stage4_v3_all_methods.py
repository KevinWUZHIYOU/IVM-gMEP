#!/usr/bin/env python3
"""Use the established Ala plotting code with the clean stage-4 v3 IVM."""

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
    source = (
        repo
        / "paper/ExperimentArchive/ala2_66d/integer_alpha20_best_20260801/plot_ala_integer20_all_methods.py"
    )
    specification = importlib.util.spec_from_file_location("legacy_ala_all_methods", source)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def load_clean_ivm(module, result_path: Path, dense_path: Path, initial_path: np.ndarray):
    saved = np.load(result_path, allow_pickle=False)
    dense = np.load(dense_path, allow_pickle=False)
    alpha = float(saved["alpha"])
    if not np.isclose(alpha, 20.0) or str(saved["alpha_mode"]) != "fixed_across_both_phases":
        raise RuntimeError("Clean stage-4 result is not fixed integer alpha=20")
    if str(saved["policy_version"]) != "stage4_uniform_v3":
        raise RuntimeError("Expected stage4_uniform_v3")
    force = np.asarray(saved["total_force_evaluations"], dtype=int)
    iperp = np.asarray(saved["i_perp"], dtype=float)
    barrier_force = np.asarray(dense["calls"], dtype=int)
    barrier = np.asarray(dense["adaptive_barrier_kcal"], dtype=float)
    if int(barrier_force[-1]) != int(force[-1]):
        raise RuntimeError("Dense diagnostic does not include the selected terminal state")
    return module.MethodSeries(
        key="ivm",
        label="IVM ($\\alpha=20$)",
        short_label="IVM",
        color="#0072B2",
        marker="^",
        linestyle="-",
        force=force,
        iperp=iperp,
        barrier_force=barrier_force,
        barrier=barrier,
        selected_call=int(saved["macro_calls"][-1]),
        selected_force=int(force[-1]),
        selected_iperp=float(iperp[-1]),
        selected_barrier=float(barrier[-1]),
        selected_path=np.asarray(saved["final_path"], dtype=float),
        initial_path=np.asarray(initial_path, dtype=float),
        path_force=np.asarray(saved["stored_force_evaluations"], dtype=int),
        paths=np.asarray(saved["stored_paths"], dtype=float),
        configuration=(
            "clean-start uniform stage-4 v3; fixed alpha=20; phase A path-sum "
            "then phase B precision resampling; online complete-macrostate stop"
        ),
        result_path=result_path,
        dense_path=dense_path,
    )


def main() -> None:
    study = Path(__file__).resolve().parents[1]
    repo = Path(__file__).resolve().parents[3]
    legacy = repo / "python/alanine_v10_chignolin_v9_fixed_alpha_20260730"
    module = load_legacy_module(repo)
    output_dir = study / "figures/clean_four_level_audit/all_methods_stage4_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ivm": study / "runs/stage4_inexact_optimized_resample/stage4_uniform_v3_a20_phaseA80_phaseB220.npz",
        "dense_ivm": study / "runs/diagnostics/dense_barrier_clean_stage4_v3_a20_complete_macrostates.npz",
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
    common_initial = np.asarray(np.load(paths["common_initial"])["initial_path"], dtype=float)
    ivm = load_clean_ivm(module, paths["ivm"], paths["dense_ivm"], common_initial)
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
    ode = module.load_ode(paths["ode"], paths["ode_metrics"], paths["dense_ode"], common_initial)
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
        method.key: float(np.max(np.abs(method.initial_path - common_initial))) for method in methods
    }
    if any(value > 5e-7 for value in initial_differences.values()):
        raise RuntimeError(f"Initial path mismatch: {initial_differences}")
    with paths["rama"].open("rb") as handle:
        rama = module._ContourCacheUnpickler(handle).load()
    module.configure_original_style()
    descent = module.plot_descent(methods, output_dir)
    cost_outputs, cost_rows, speedups = module.plot_cost_and_paths(methods, rama, output_dir)
    evolution = module.plot_path_evolution(methods, rama, output_dir)
    summary_path = output_dir / "alanine_stage4_v3_selected_methods.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "selected_NF", "selected_i_perp", "selected_dense_barrier", "barrier_abs_error"])
        for method in methods:
            writer.writerow([
                method.short_label, method.selected_force, method.selected_iperp,
                method.selected_barrier, abs(method.selected_barrier - module.REFERENCE_BARRIER_KCAL)
            ])
    audit = {
        "ivm_policy": "stage4_uniform_v3",
        "ivm_alpha": 20,
        "ivm_selected_NF": ivm.selected_force,
        "ivm_selected_i_perp": ivm.selected_iperp,
        "ivm_selected_dense_barrier": ivm.selected_barrier,
        "ivm_barrier_abs_error": abs(ivm.selected_barrier - module.REFERENCE_BARRIER_KCAL),
        "ode_terminal_i_perp": ode.selected_iperp,
        "ode_terminal_barrier_abs_error": abs(ode.selected_barrier - module.REFERENCE_BARRIER_KCAL),
        "ivm_beats_ode_i_perp": bool(ivm.selected_iperp < ode.selected_iperp),
        "ivm_beats_ode_barrier_error": bool(abs(ivm.selected_barrier - module.REFERENCE_BARRIER_KCAL) < abs(ode.selected_barrier - module.REFERENCE_BARRIER_KCAL)),
        "first_passage_speedups": speedups,
        "initial_path_max_abs_differences": initial_differences,
        "outputs": {"descent": descent, "cost_and_paths": cost_outputs, "evolution": evolution},
        "cost_rows": cost_rows,
    }
    (output_dir / "alanine_stage4_v3_all_methods_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

