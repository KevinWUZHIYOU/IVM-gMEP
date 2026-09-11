# Experiment, code, data, and figure index

All paths in this table are relative to this study directory unless they are
absolute. Main-result counts exclude smoke tests and tuning pilots.

| experiment | main data | runner / postprocessor | formal figure or report | status |
| --- | --- | --- | --- | --- |
| Stage 1: exact-like, no resampling, `r=0.2/0.1` | `runs/stage1_exact_no_resample/` (20 NPZ) and matching dense files in `runs/diagnostics/` | `src/run_clean_ivm_grid.py`, `src/run_ivm_v9_spring_projected_production_20260730.py`, `src/postprocess_dense_barrier.py` | `figures/clean_four_level_audit/alanine_clean_exact_alpha_two_metric_2x4.{pdf,svg,png}` | complete, fail-closed audit passed |
| Stage 2: exact-like + resampling, `r=0.2/0.1` | `runs/stage2_exact_resample/` (20 NPZ) and matching dense files in `runs/diagnostics/` | same as Stage 1 | same 2x4 figure | complete, fail-closed audit passed |
| Stage 3: baseline inexact + resampling | `runs/stage3_inexact_baseline_resample/` (10 NPZ) and matching dense files | same core runner and dense postprocessor | `figures/clean_four_level_audit/alanine_clean_inexact_alpha_two_metric_2x2.{pdf,svg,png}` | complete, fail-closed audit passed |
| Stage 4: uniform optimized inexact + resampling | `runs/stage4_inexact_optimized_resample/stage4_uniform_v3_a*_phaseA34_phaseB220.npz` (10 NPZ) | `src/run_stage4_uniform_optimized.py`, core runner, dense postprocessor | baseline 1x2: `figures/clean_four_level_audit/alanine_clean_stage4_alpha_cost_to_precision_1x2.{pdf,svg,png}` | complete |
| Stage 4 v4 accepted-state archive | `runs/pareto_archive/pareto_clean_stage4_v4_a*_phaseA34_phaseB220.{npz,audit.json}` (10 pairs) | `src/build_pareto_archive.py`, `src/audit_stage4_v4_acceptance.py` | `reports/stage4_v4_fail_closed_acceptance.json` | accepted; only integer alpha 20 reaches target |
| Alpha cost-to-precision, common 5000-force decade grid | Stage-1 `r=0.1` long runs and uniformly extended Stage-4 runs/archives, all truncated at charged `N_F=5000` | `src/plot_alpha_common5000_decades_retitle_v10.py` | `figures/clean_four_level_audit/alpha_common5000_decades_v10_titles/alanine_v10_cut5000_decades_protocol_titles_alpha_budget_aware_cost_to_precision_stage1_vs_stage4.{pdf,svg,png}` | final paper figure; protocol titles audited |
| All-method two-metric descent | selected alpha-20 archive plus frozen Original String, static NEB, ODE-NEB, OBSM-BFGS, OBSM-FIRE data referenced in the figure audit JSON | `src/plot_all_methods_linear_speedup_v5.py` using the established v4 loader | `figures/clean_four_level_audit/all_methods_stage4_v4_pareto_linear_speedup_v5/alanine_integer20_all_methods_two_metric_descent_linear_cost_speedup.{pdf,svg,png}` | final paper figure; linear x axis, 17.4x/4.0x annotations |
| All-method cost-to-precision + selected paths | same data as previous row | same plotting source | `.../alanine_integer20_all_methods_cost_to_precision_and_paths.{pdf,svg,png}` | final v4 figure |
| IVM 1x4 path evolution | exact stored accepted alpha-20 states; selected path SHA-256 `009b6d129797a54b02f6c7424426ea24c76367a184f83e0066bfbcb9fae0bcff` | `src/plot_stage4_v4_pareto_all_methods.py` | `figures/clean_four_level_audit/all_methods_stage4_v4_pareto/alanine_integer20_all_methods_1x4_path_evolution.{pdf,svg,png}` | final paper figure |
| Four protocol-separated audit figures | all Stage 1/2/3 trajectories and diagnostics, all ten A34/B220 Stage-4 runs and v4 archives | `src/plot_clean_four_groups_clear_v4.py` | `figures/clean_four_level_audit/clear_four_groups_v4/alanine_clear_stage{1,2,3,4}_*` plus `alanine_clear_four_level_alpha20_overview.{pdf,svg,png}` | clear v4 audit set; Stage IV raw and accepted traces explicitly separated |

Common input: `inputs/common_initial_path_equidistant.npz`; topology:
`inputs/ala2_fixed.pdb`.

The exact source-to-file competitor mapping, selected checkpoints, metrics,
and references are in
`figures/clean_four_level_audit/all_methods_stage4_v4_pareto/alanine_stage4_v4_pareto_all_methods_audit.json`.
The selected curve values are also exported as
`alanine_stage4_v4_pareto_selected_methods.csv` in that directory.

Rejected precision/restart/CV-trigger pilots are retained under
`runs/stage4_policy_pilots/`, `runs/diagnostics/`, and
`reports/stage4_phaseb_policy_pilots*.json`; they are not silently mixed into
the selected alpha grid.

Pre-audit archives remain frozen and are not overwritten:

- integer-alpha20 baseline experiment:
  `/Users/wuzhiyou/IVM_for_MEP/paper/ExperimentArchive/ala2_66d/integer_alpha20_best_20260801/`;
- clean v2 experiment:
  `/Users/wuzhiyou/IVM_for_MEP/paper/ExperimentArchive/ala2_66d/clean_four_level_audit_v2_20260801/`;
- corresponding figure directories under
  `/Users/wuzhiyou/IVM_for_MEP/paper/Figures/ala2_66d/`.
