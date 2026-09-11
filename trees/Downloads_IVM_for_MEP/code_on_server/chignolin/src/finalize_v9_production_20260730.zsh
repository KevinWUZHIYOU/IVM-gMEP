#!/bin/zsh
set -euo pipefail

experiment_root=/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin
analysis_root="$experiment_root/src"
python_executable=/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python3.8
grid_report="$experiment_root/reports/v9_kkt0p7_cap30_final_grid_20260730"
dense_report="$experiment_root/reports/v9_kkt0p7_cap30_final_dense_20260730"
selection_report="$experiment_root/reports/v9_kkt0p7_cap30_final_selection_20260730"
figure_directory=figures/article_update_20260730_v9_final
alpha_figure_root="$experiment_root/$figure_directory"
method_summary="$experiment_root/figures/article_update_20260728e/chignolin_method_comparison_summary.json"
alpha_one_result="$experiment_root/runs/ivm/ivm_fixed_alpha_v9_kkt0p7_cap30_a1_central_state_ivm_cfg9b5a238786a3_budget650.npz"
alpha_one_dense="$experiment_root/runs/diagnostics/dense_barrier_v9_kkt0p7_cap30_anytime_grid_20260730_a1.npz"
export MPLCONFIGDIR=/tmp/chignolin_v9_mplconfig_20260730
mkdir -p "$MPLCONFIGDIR"

"$python_executable" \
  "$analysis_root/summarize_v9_kkt0p7_cap30_grid_20260730.py" \
  --root "$experiment_root" \
  --output-dir "$grid_report"

env \
  OPENMM_CPU_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 \
  PYTHONPATH="$experiment_root/src" \
  "$python_executable" \
  "$analysis_root/run_v9_dense_alpha_diagnostics_20260730.py" \
  --root "$experiment_root" \
  --grid-summary "$grid_report/v9_kkt0p7_cap30_grid_summary.json" \
  --python "$python_executable" \
  --output-dir "$dense_report" \
  --output-label-prefix v9_kkt0p7_cap30_anytime_grid_20260730 \
  --outer-workers 2 \
  --diagnostic-workers 2

"$python_executable" \
  "$analysis_root/score_v9_alpha_final_window_20260730.py" \
  --grid-summary "$grid_report/v9_kkt0p7_cap30_grid_summary.json" \
  --dense-manifest "$dense_report/dense_manifest.json" \
  --method-summary "$method_summary" \
  --analysis-contract \
  "$analysis_root/V9_FINAL_WINDOW_ANALYSIS_CONTRACT_20260730.md" \
  --output-dir "$selection_report"

"$python_executable" \
  "$analysis_root/plot_v9_alpha_final_window_well_20260730.py" \
  --grid-summary "$grid_report/v9_kkt0p7_cap30_grid_summary.json" \
  --dense-manifest "$dense_report/dense_manifest.json" \
  --output-dir "$alpha_figure_root"

env \
  PYTHONPATH="$experiment_root/src" \
  "$python_executable" \
  "$analysis_root/plot_paper_method_comparison_v3_total_nf_20260730.py" \
  --root "$experiment_root" \
  --output-directory "$figure_directory" \
  --ivm-result "$alpha_one_result" \
  --ivm-dense-result "$alpha_one_dense" \
  --ivm-selection-report \
  "$selection_report/V9_ALPHA_FINAL_WINDOW_SCORES.json"
