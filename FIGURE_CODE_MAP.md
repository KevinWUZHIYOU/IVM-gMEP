# Figure → code map

All 16 figures that `paper0819/main.tex` actually typesets, in manuscript
order. Line numbers refer to `main.tex`.

Path aliases used below, relative to this release root:

- `$D` = `trees/Downloads_IVM_for_MEP`
- `$I` = `trees/IVM_for_MEP`
- `$C` = `trees/Code/IVM/MFEP_ala/amber14sb_vacuo`

## Summary

| # | LaTeX label | Figure PDF | Plotting code | Experiment code |
| --- | --- | --- | --- | --- |
| 1 | `fig:LEPS` (l.1610) | `$D/paper0819/Figures/LEPS/leps_path_evolution_1x4.pdf` | `$D/code_on_server/LEPS/leps_paper_update_20260906/src/update_paper_leps_figures.py` | `$D/code_on_server/LEPS/leps_irivm_sync1_refined_grid_20260830/`, `$D/code_on_server/LEPS/leps_irivm_ala_chig_every1_sync1_20260830/src/leps_irivm_sync1.py`, `$D/code_on_server/LEPS/src/` |
| 2 | `fig:LEPS_all_methods` (l.1630) | `$D/paper0819/Figures/LEPS/leps_all_methods_metrics.pdf` | `$D/code_on_server/LEPS/leps_action_baselines_20260910/analyze_and_plot.py` | `$D/code_on_server/LEPS/leps_action_baselines_20260910/` (runner + registration), `$D/code_on_server/LEPS/src/`, `$D/code_on_server/LEPS/leps_irivm_ala_chig_lepsnative_20260829/` |
| 3 | `fig:LEPS_continuous` (l.1698) | `$D/paper0819/Figures/LEPS/leps_continuous_methods_comparison.pdf` | `$D/code_on_server/LEPS/leps_paper_update_20260906/src/update_paper_leps_figures.py` | `$D/code_on_server/LEPS/src/` (NN-IVM, neural gMAM, ODE–NEB, continuous-path metrics) |
| 4 | `fig:M10_density` (l.1741) | `$D/paper0819/Figures/Muller10/muller_q10_article_method_comparison_n10_n20_2x4_geometric_action_linear3000.pdf` | `$D/paper0819/Figures/Muller10/plot_muller_q10_article_method_comparison_n10_n20_2x3_no_action.py` | `$I/python/q10_irivm_recovery_monotone_exploration_20260831/src/`, `$I/python/q10_irivm_low_r_coherence_secant_v2_20260830/src/`, `$I/python/q10_irivm_unified_cycle_frozen_reproduction_v2_20260831/src/` |
| 5 | `fig:Ala2_accuracy` (l.1847) | `$D/paper0819/Figures/ala2_66d/alanine66d_grid_ivm_path_evolution_1x4.pdf` | `$I/paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_grid_ivm_path_evolution_1x4.py` | `$I/python/alanine_ivm_clean_four_level_20260801/` |
| 6 | `fig:Ala2_convergence` (l.1947) | `$D/paper0819/Figures/ala2_66d/alanine66d_ivm_geometric_action_1x3.pdf` | `$I/paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_delta_aware_geometric_action_1x3.py` | `$I/paper/ActiveExperiments/alanine_66d/code/metrics/postprocess_ala_geometric_action.py`, `$I/python/alanine_ivm_clean_four_level_20260801/` |
| 7 | `fig:Ala2_configurations` (l.1980) | `$D/paper0819/Figures/ala2_66d/alanine66d_energy_and_configurations.pdf` | `$I/paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_energy_and_configurations.py`, `.../render_ala_ballstick.py` | `$I/python/alanine_ivm_clean_four_level_20260801/`, `$I/python/alanine_v10_chignolin_v9_fixed_alpha_20260730/src/plot_v49_formal_ala_1x3.py` |
| 8 | `fig:Chignolin_transition` (l.2035) | `$D/paper0819/Figures/chignolin/transition_2x5.pdf` | **not found** — see "Known gaps" | **not found** |
| 9 | `fig:Chignolin_convergence` (l.2203) | `$D/paper0819/Figures/chignolin/chignolin_ir_ivm_overlay_total_nf.pdf` | `$I/python/chignolin_ir_ivm_20260826/src/plot_chignolin_ivm_three_metrics.py` | `$I/python/chignolin_ir_ivm_20260826/src/`, `$D/code_on_server/chignolin/src/` |
| 10 | `fig:barnase-barstar-pes` (l.2550) | `$D/paper0819/Figures/barnase_barstar/barnase_barstar_ir_ivm_evolution.pdf` | `$D/code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/plot_bb_r2_smooth_c2_gamma15_to_linear_preview.py`, `.../plot_bb_r2_merged_relaxed15x10_evolution.py` | `$I/python/barnase_barstar_ir_ivm_20260901/`, `$D/code_on_server/bb_neb_aseode30_kgrid_nf6000_nf30000_20260903_r1/src/` |
| 11 | `fig:barnase-barstar-cost` (l.2575) | `$D/paper0819/Figures/barnase_barstar/barnase_barstar_ir_ivm_cost_to_precision.pdf` | `$D/code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/plot_bb_irivm20k_ode30k_cost_to_precision_1x3.py` | `$I/python/barnase_barstar_ir_ivm_20260901/`, `$D/code_on_server/bb_neb_aseode30_kgrid_nf6000_nf30000_20260903_r1/src/` |
| 12 | `fig:barnase-barstar-snapshots` (l.2606) | `$D/paper0819/Figures/barnase_barstar/barnase_barstar_ir_ivm_energy_and_configurations.pdf` | `$D/code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/plot_bb_irivm20k_ode30k_snapshots_energy.py` | `$I/python/barnase_barstar_ir_ivm_20260901/` |
| 13 | `fig:ala2_MFEP_contour` (l.2785) | `$D/paper0819/Figures/ala2_2d_FreeEnergy/MFEP_IVM_vs_FENEB_contour_fourpanel.pdf` | `$C/figure.ipynb` | `$C/IVM_shell.py`, `$C/feneb_core.py`, `$C/FENEB_ala_compare.py`, `$C/IVM_LBFGS_warmup_shell.ipynb` |
| 14 | `fig:ala2_MFEP_accuracy` (l.2798) | `$D/paper0819/Figures/ala2_2d_FreeEnergy/MFEP_IVM_vs_FENEB_bestpair_threepanel.pdf` | `$C/figure.ipynb` | `$C/IVM_shell.py`, `$C/feneb_core.py`, `$C/FENEB_ala_compare.py`, `$C/IVM_LBFGS_warmup_shell.ipynb` |
| 15 | `fig:MFEP_3paths_comparison_accuracy` (l.2833) | `$D/paper0819/Figures/ala2_2d_FreeEnergy/MFEP_vs_CV_MEP_vs_projected_66D_MEP_normalized_progress_metrics_and_contour.pdf` | `$C/figure.ipynb` | `$C/IVM_shell.py`, `$C/ase_mfep_neb.py`, `$C/ALA_NEB_ODE.py`, `$C/optimal_IVM.ipynb` |
| 16 | `fig:MFEP_3paths_comparison_contour` (l.2848) | `$D/paper0819/Figures/ala2_2d_FreeEnergy/MFEP_CVMEP_66DMEP_full_and_zoom_contour.pdf` | `$C/figure.ipynb` | `$C/IVM_shell.py`, `$C/ase_mfep_neb.py`, `$C/ALA_NEB_ODE.py` |

## Per-figure provenance records

| # | Provenance record |
| --- | --- |
| 1–3 | `$D/code_on_server/LEPS/leps_paper_update_20260906/LEPS_PAPER_UPDATE_PROVENANCE.md`, `$D/code_on_server/LEPS/leps_irivm_sync1_refined_grid_20260830/FIGURE_PROVENANCE.md`, `$D/code_on_server/LEPS/leps_action_baselines_20260910/README.md` + `reports/leps_all_methods_metrics_provenance.json` |
| 4 | `$D/paper0819/Figures/Muller10/muller_q10_article_method_comparison_n10_n20_2x4_geometric_action_linear3000_PROVENANCE.md` + `_manifest.json` (references `$I/paper/ActiveExperiments/muller_q10_10d/`) |
| 5–7 | `$D/paper0819/Figures/ala2_66d/FIGURE_GRID_IVM_PATH_EVOLUTION_1X4_PROVENANCE.md`, `alanine66d_ivm_geometric_action_1x3_provenance.md`, `FIGURE_ENERGY_AND_CONFIGURATIONS_PROVENANCE.md` + `_manifest.json` |
| 8 | none located |
| 9 | `$D/paper0819/Figures/chignolin/chignolin_ir_ivm_overlay_total_nf_SOURCE.md` + `_provenance.json`, `$I/python/chignolin_ir_ivm_20260826/RESULTS_AND_PROVENANCE.md` |
| 10–12 | `$D/paper0819/Figures/barnase_barstar/formal_ir_ivm_20260905/*.provenance.json`, `*.traceability.md`, `BB_IR_IVM_FORMAL_EXPERIMENT_20260905.md` |
| 13–16 | No separate registration document was found for these four PDFs. Their provenance is carried inside `$C/figure.ipynb` itself, which pins `BASE_PATH`, an explicit input-file list (including the metadynamics-generated restraint cache and the CV-space MFEP/MEP result files) and one `OUTPUT_STEM` per figure immediately above each plotting section. |

## Shared and supporting code

Several figures draw on the same underlying libraries and diagnostics:

- **LEPS potential and NEB baselines** – `$D/code_on_server/LEPS/src/` holds the
  analytical potential, the independent NumPy energy/force checks, the ASE
  ODE–NEB baseline, the NN-IVM and neural gMAM implementations, the common
  ASE-IDPP initial path generator, and the metric analyzers used by figures
  1–3.
- **LEPS registered runs** – `leps_irivm_sync1_refined_grid_20260830/runs/` and
  `leps_action_baselines_20260910/runs/` contain the raw `.npz` states behind
  the plotted curves, and `registrations/` plus `reports/selection.json`
  freeze the parameter choices.
- **Muller QM10 core** – `$I/paper/ActiveExperiments/muller_q10_10d/code/`
  supplies the QM10 potential, the improved-tangent core and the baseline
  loaders that the figure-4 plotting script imports; the selected IVM curves
  themselves come from the `q10_irivm_*` campaigns under `$I/python/`.
- **Alanine 66D** – `$I/paper/ActiveExperiments/alanine_66d/README_FIRST.md`
  and `SHA256SUMS` describe the frozen data registry; the executables are in
  `$I/python/alanine_ivm_clean_four_level_20260801/src/`.
- **Chignolin** – `$I/python/chignolin_ir_ivm_20260826/src/` holds the
  IR-IVM optimizer path and the three-metric plotter plus its arc-length-500
  and discrete-geometric-action post-processing; `$D/code_on_server/chignolin/`
  holds the earlier V9 production runner, the ODE/static NEB and OBSM
  baselines, and the frozen NEB production `.npz` files that the overlay reads.
- **Barnase–barstar** – `$I/python/barnase_barstar_ir_ivm_20260901/` holds the
  IR-IVM source, the full-state history replays and the reports that the three
  figure provers authenticate; the ODE–NEB side and the plotters live in
  `$D/code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/`.
- **Alanine 2D MFEP** – `$C` is the original experiment directory: `IVM_shell.py`
  and the `IVM*` notebooks are the IVM side, `feneb_core.py` /
  `FENEB_ala_compare.py` / `ase_mfep_neb.py` the FENEB and NEB comparisons, and
  `figure.ipynb` the notebook that renders all four MFEP PDFs. The `.npz`
  summaries it reads are included alongside.

## Figures deliberately not included

`main.tex` keeps several older Chignolin PDFs inside `\iffalse ... \fi` blocks
(`chignolin_fixed_alpha1_metrics.pdf`,
`chignolin_fixed_alpha1_cost_to_precision.pdf`,
`chignolin_reparameterization_ablation.pdf`). They are not typeset and are
therefore not part of the formal figure set mapped above. Their plotters happen
to sit in the same source directories as the active Chignolin plotter
(`$D/code_on_server/chignolin/src/plot_fixed_alpha1_comparison.py` and
`$I/python/chignolin_ir_ivm_20260826/src/plot_chignolin_geometric_action.py`),
so those files are present in this release as supporting context rather than as
formal-figure code.
