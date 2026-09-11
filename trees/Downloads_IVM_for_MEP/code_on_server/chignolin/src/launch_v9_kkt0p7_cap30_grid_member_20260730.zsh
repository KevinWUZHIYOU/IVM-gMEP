#!/bin/zsh
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  print -u2 "usage: $0 alpha [budget_per_image]"
  exit 2
fi

alpha=$1
budget=${2:-650}
alpha_token=${alpha//./p}
experiment_root=/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin
python_executable=/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python
runner="$experiment_root/src/run_ivm_v9_spring_projected_production_20260730.py"
run_label="fixed_alpha_v9_kkt0p7_cap30_a${alpha_token}"
log_dir="$experiment_root/logs/v9_kkt0p7_cap30_grid_20260730"

mkdir -p "$log_dir"
export OPENMM_CPU_THREADS=${OPENMM_CPU_THREADS:-2}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

existing=(
  "$experiment_root"/runs/ivm/ivm_${run_label}_central_state_ivm_*_budget${budget}.npz(N)
)
if (( ${#existing} > 0 )); then
  print "SKIP ${run_label}: ${existing[1]}"
  exit 0
fi

"$python_executable" "$runner" \
  --root "$experiment_root" \
  --initial-path-file inputs/common_initial_path_equidistant.npz \
  --run-label "$run_label" \
  --solver state_ivm \
  --alpha "$alpha" \
  --alpha-mode fixed \
  --alpha-min 0.03 \
  --alpha-max 20 \
  --max-step 1 \
  --state-trust-fraction 2 \
  --state-polish-trust-fraction 2 \
  --state-globalization proximal_only \
  --reparam-mode state_curve_projected \
  --state-curve-direct-solve \
  --state-curve-direct-trial-calls 30 \
  --state-curve-immediate-block-relock \
  --state-curve-relock-escape-trust-factor 1 \
  --state-curve-stall-free-polish \
  --state-curve-stall-free-polish-count 1 \
  --state-free-polish-reparam-strategy relative_defect \
  --state-free-polish-relative-defect-trigger 5e-7 \
  --state-free-polish-relative-defect-target 2.5e-7 \
  --reparam-spring-dtau 0.05 \
  --state-split-dtau-max 0.5 \
  --reparam-ready-relative-iperp 2 \
  --reparam-cv-trigger -1 \
  --reparam-recovery-calls 30 \
  --plbfgs-kkt-rtol 0.7 \
  --inner-calls 30 \
  --budget "$budget" \
  --save-all-paths \
  >"$log_dir/${run_label}_budget${budget}.log" 2>&1
