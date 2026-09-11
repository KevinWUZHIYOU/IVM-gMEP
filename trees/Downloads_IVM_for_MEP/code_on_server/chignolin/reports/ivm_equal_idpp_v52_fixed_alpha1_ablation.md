# Chignolin v52 fixed-alpha1 ablation

## Status

This run is a preserved ablation and does **not** replace the selected adaptive
v52 production result.

The only intended algorithmic change from v52 is

```text
--alpha-mode adaptive  ->  --alpha-mode fixed
```

with `--alpha 1`.  The common initial path, central-difference tangent,
state-IVM solver, force-field preconditioner, central-Hermite equal-spring
projection, trust-radius rule, KKT forcing tolerance, and all other controls
are unchanged.

## Exact command

```bash
cd /Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
OPENMM_CPU_THREADS=1 \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/run_ivm.py \
  --root /Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin \
  --initial-path-file inputs/common_initial_path_equidistant.npz \
  --run-label ivm_equal_idpp_v52_fixed_alpha1_production \
  --solver state_ivm \
  --alpha 1 \
  --alpha-mode fixed \
  --alpha-min 0.03 \
  --alpha-max 20 \
  --max-step 1 \
  --state-trust-fraction 2 \
  --state-polish-trust-fraction 2 \
  --reparam-mode state_curve_projected \
  --state-curve-direct-solve \
  --state-curve-direct-trial-calls 15 \
  --reparam-ready-relative-iperp 2 \
  --reparam-cv-trigger -1 \
  --reparam-recovery-calls 30 \
  --inner-calls 15 \
  --plbfgs-kkt-rtol 0.8 \
  --budget 500 \
  --save-all-paths
```

## Outcome

The run stopped honestly at call 342 with `lbfgsb_no_progress`; it was not
padded to 500 calls.  The last accepted path occurred at call 324.

- fixed \(\alpha=1\) throughout;
- final unnormalized \(I_\perp=0.03885242073674519\);
- final sparse barrier \(=19.69329514119104\) kcal/mol;
- final physical normal \(f_{\max}=0.008062898867391447\) eV/Å;
- final equidistance integral \(=5.885353699911007\times10^{-20}\);
- final segment-length CV \(=6.931057213625601\times10^{-16}\);
- independently densified barrier \(=20.373739515742795\) kcal/mol, inherited
  from the adaptive-v52 call-325 diagnostic because that stored coordinate is
  byte-identical to the fixed run's final path.

At the fixed run's final \(I_\perp\) threshold, IVM first arrives at call 324
and the selected \(k=50\) ODE-NEB first arrives at call 4082, corresponding to
a \(12.5988\times\) first-passage speedup.

## Exact comparison with adaptive v52

The fixed and adaptive trajectories are byte-identical for calls 1--342:

- stored coordinates: exact equality, maximum absolute difference 0;
- \(I_\perp\): exact equality;
- sparse barrier: exact equality;
- physical normal \(f_{\max}\): exact equality;
- equidistance integral and segment-length CV: exact equality.

The selected adaptive v52 subsequently reduces \(\alpha\) to 0.5 and accepts
one additional update at call 344:

| metric | fixed alpha1 | selected adaptive v52 |
|---|---:|---:|
| last accepted call | 324 | 344 |
| final \(I_\perp\) | 0.0388524207 | 0.0381150078 |
| sparse barrier (kcal/mol) | 19.6932951 | 19.6904470 |
| densified barrier (kcal/mol) | 20.3737395 | 20.3754058 |
| physical \(f_{\max}\) (eV/Å) | 0.00806290 | 0.00789048 |

Thus fixed \(\alpha=1\) preserves the entire early and middle performance and
most of the final accuracy.  Adaptive shrinkage provides a small but genuine
late \(I_\perp\)/\(f_{\max}\) improvement.  The fixed run should be presented
as a clean hyperparameter ablation, not silently substituted for the selected
v52 production trajectory.

## Force-call audit

- records: 342;
- internal-image joint evaluations min/mean/max/sum:
  \(341/341.05/342/6821\);
- total 22-image evaluator cache misses: 6823;
- backend raw OpenMM joint evaluations: 6823;
- unchanged-path intervals containing real rejected evaluations: 19;
- audit result: passed.

## Integrity

- fixed-alpha1 NPZ SHA-256:
  `7f99e96d1337283fc2361752e16ccfac1bd9679b085ea099f9268d2de19267d7`
- selected adaptive-v52 NPZ SHA-256:
  `99075b36636c03f42d9184c3c54df1a2b7eb930b328e91e643f0359db8a75cdd`
- selected adaptive-v52 manifest SHA-256:
  `fb5400c66de86434b8e0ae3797c81077cb6ac07ffe8bab2386e447fb39c6f63a`

