# Chignolin fixed-alpha1 presentation and reparameterization ablation

## Status

The fixed-\(\alpha=1\) equal-spring v52 run is now the selected presentation
curve. The earlier adaptive-alpha v52 production run is preserved unchanged
as a fallback and provenance record.

All curves use the same equal-IDPP initial path
(`57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996`)
and the same normalized central-difference tangent. No CI is used.

## Selected fixed-alpha1 result

The selected presentation run uses the certified state-IVM subminimization,
fixed \(\alpha=1\), and the central-Hermite equal-spring gauge. It stopped
honestly at call 342 with `lbfgsb_no_progress`; the final accepted coordinate
was reached at call 324. The curve is not extended horizontally to call 500.

- \(I_\perp=0.03885242073674519\);
- sparse barrier \(=19.69329514119104\) kcal/mol;
- independently densified barrier at the final coordinate
  \(=20.373739515742795\) kcal/mol;
- best observed densified barrier \(=20.35667037607208\) kcal/mol at call 300;
- physical normal \(f_{\max}=0.008062898867391447\) eV/Å;
- equidistance integral \(=5.885353699911007\times10^{-20}\);
- segment-length CV \(=6.931057213625601\times10^{-16}\).

At the strictest common thresholds in the new cost-to-precision plot:

- \(I_\perp=0.03885242073674519\): IVM 324 calls, NEB 4082 calls,
  \(12.60\times\);
- densified barrier \(=20.407641427938866\) kcal/mol: IVM 300 calls,
  NEB 4250 calls, \(14.17\times\).

## Strict no-reparameterization ablation

This is a new controlled ablation, not a relabeling of an old run. It keeps the
same initial path, tangent, state-IVM subproblem, force-field preconditioner,
KKT tolerance, trust controls, and fixed \(\alpha=1\), but sets
`--reparam-every 0`.

The saved trajectory confirms:

- one unique active alpha, exactly 1;
- `state_reparam_attempts = 0`;
- `state_reparam_accepts = 0`;
- zero explicit reparameterization-attempt records;
- zero explicit reparameterization-commit records.

Exact command:

```bash
cd /Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
OPENMM_CPU_THREADS=1 \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/run_ivm.py \
  --root /Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin \
  --initial-path-file inputs/common_initial_path_equidistant.npz \
  --run-label ivm_equal_idpp_v52_fixed_alpha1_no_reparam_ablation \
  --solver state_ivm \
  --alpha 1 \
  --alpha-mode fixed \
  --alpha-min 0.03 \
  --alpha-max 20 \
  --max-step 1 \
  --state-trust-fraction 2 \
  --state-polish-trust-fraction 2 \
  --reparam-mode geometric \
  --reparam-every 0 \
  --reparam-ready-relative-iperp 2 \
  --reparam-cv-trigger -1 \
  --reparam-recovery-calls 30 \
  --inner-calls 15 \
  --plbfgs-kkt-rtol 0.8 \
  --budget 500 \
  --save-all-paths
```

At call 500 it reaches:

- \(I_\perp=0.025128301825040653\);
- sparse barrier \(=19.299509616726013\) kcal/mol;
- physical normal \(f_{\max}=0.006268762450842394\) eV/Å;
- segment-length CV \(=0.11259311028277966\);
- segment-length ratio \(=1.5170219300855774\);
- equidistance integral \(=8.31963419209975\times10^8\).

This result demonstrates that the IVM subminimization can converge without
reparameterization. It also exposes the finite-image discretization tradeoff:
the densified barrier improves to \(22.68695069680342\) kcal/mol at call 325,
then rises to \(35.36217322393559\) kcal/mol at call 500 even while the
image-point \(I_\perp\) and sparse barrier continue to improve. Therefore the
strict no-reparameterization trajectory is shown as an ablation, not as a
second production-quality barrier curve. Its best attainable densified-barrier
threshold is reached in 325 calls versus 1950 for NEB, or \(6.00\times\).

## Historical v46 clarification

The remembered pre-v52 run is not strictly free of reparameterization. It uses
an adaptive-alpha mode whose realized alpha nevertheless remained exactly 1,
attempted the state reparameterization 19 times, and accepted it 11 times.
It is therefore labeled **sparse reparameterization**, not **no
reparameterization**. Its final densified barrier is
\(21.026571299324658\) kcal/mol at call 500. It is included as a historical
reference only, because it is not a one-factor controlled ablation of v52.

## Call audit

The strict no-reparameterization run passed both selected-run and force-call
audits:

- internal-image joint evaluations min/mean/max/sum:
  \(443/446.55/500/8931\);
- total 22-image evaluator cache misses: 8933;
- backend raw OpenMM joint evaluations: 8933;
- 52 unchanged saved-path intervals contain genuine evaluated-and-rejected
  trials, proving rollback calls were not erased.

Every cache miss jointly obtained energy and force. The densified-barrier
diagnostic used separate evaluators and its calls are excluded from the
optimization horizontal axis.

## Figures

The new output directory is `figures/equal_idpp_fixed_alpha1_selected`.

- `chignolin_fixed_alpha1_metrics`: fixed-alpha1 equal-spring IVM, strict
  no-reparameterization IVM, historical sparse-reparameterization v46, and
  ODE-NEB;
- `chignolin_fixed_alpha1_cost_to_precision`: fixed-alpha1 IVM and NEB over
  their full common precision range; the no-reparameterization curve stops
  where it no longer attains the densified-barrier threshold;
- `chignolin_reparameterization_ablation`: \(I_\perp\), equidistance integral,
  and segment-length CV for the three IVM variants.

PDF, SVG, 600-dpi PNG, CSV, and JSON outputs are saved. The new plotter rejects
unequal initial-path hashes, rejects a non-fixed-alpha main run, and rejects
any purported no-reparameterization run containing a reparameterization
attempt or commit.

## Verification and preservation

- all 48 automated tests passed;
- the fixed-\(\alpha=1\) result, strict no-reparameterization result, dense
  diagnostics, plotter, figures, audits, and source hashes are recorded in
  `snapshots/equal_idpp_fixed_alpha1_selected/manifest.json`;
- the adaptive v52 result remains byte-for-byte unchanged at SHA-256
  `99075b36636c03f42d9184c3c54df1a2b7eb930b328e91e643f0359db8a75cdd`;
- its manifest remains unchanged at SHA-256
  `fb5400c66de86434b8e0ae3797c81077cb6ac07ffe8bab2386e447fb39c6f63a`.
