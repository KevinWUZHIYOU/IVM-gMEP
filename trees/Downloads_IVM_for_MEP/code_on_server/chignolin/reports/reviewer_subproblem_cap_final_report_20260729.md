# Chignolin fixed-\(\alpha\) subproblem-cap and double-well audit

Date: 2026-07-29

## Decision

The cap-18 and cap-20 production-style sweeps are complete, but the
preregistered requirement of a stable, single alpha well was **not met**.
Increasing the frozen-tangent subproblem cap changes the large-alpha
transients, but does not remove the rugged multi-well structure of raw
\(I_\perp\) versus alpha.  Therefore the existing cap-15, alpha-1 production
run remains the paper production run.  It must not be replaced merely because
cap 20 ranks first under the auxiliary preregistered aggregate score.

This is not a performance loss: for alpha 1, cap 15, 18, and 20 have identical
paths and identical \(I_\perp\) at every shared production checkpoint through
\(N_F=6000\).  The cap-20 final path is bitwise identical to the existing
production final path and has the same final \(I_\perp=0.03885242073674519\).
The larger cap only pays extra terminal failed/probe evaluations
(\(N_F=6923\) instead of 6823) without changing the final path.

## Controlled production experiment

- Caps: 15, 18, 20.
- Fixed alphas:
  \(0.03,0.05,0.1,0.2,0.3,0.5,1,2,3,5,10,15,20,30,50,70,100,300,1000\).
- 57/57 unique optimization runs passed the strict audit.
- Common initial-path SHA-256:
  `57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996`.
- Tangent: normalized central difference
  \(x_{i+1}-x_{i-1}\).
- Globalization: proximal objective only.
- Common relative KKT tolerance: 0.8.
- Cost: `counter_vectors.sum(axis=1)`, i.e. all joint energy-force cache
  misses over all 22 images.
- No smoothing, regression, interpolation, or terminal-value carry-forward.

At alpha 1 the three caps give exactly:

| Total-force budget | Endpoint \(I_\perp\) | trailing-30% mean \(I_\perp\) |
|---:|---:|---:|
| 1,000 | 0.3332032837 | 0.4339267 |
| 3,000 | 0.0978089 | 0.115751 |
| 6,000 | 0.046381 | 0.054071 |

The real 0 K OpenMM densified barriers at the same alpha are also identical:

| Total-force budget | attained \(N_F\) | barrier (kcal mol\(^{-1}\)) |
|---:|---:|---:|
| 1,000 | 982 | 24.8296713855 |
| 3,000 | 2,982 | 22.7448051580 |
| 6,000 | 5,982 | 20.3670673337 |

None of the alpha-1 runs reached \(N_F=10000\), so that cell is now a gap.
An older cap-15 diagnostic had incorrectly mapped its terminal
\(N_F=6823\) state to the 10,000 budget.  Cap 15 was recomputed with the new
strict postprocessor, and the final figure no longer carries that value
forward.

## Real OpenMM barrier audit

- Cap 15: 19/19 diagnostic runs complete; 26,435 diagnostic evaluations.
- Caps 18 and 20: 38/38 diagnostic runs complete; 54,965 diagnostic
  evaluations.
- Total new strict diagnostic evaluations: 81,400.
- Protocol: 10 and 20 subdivisions per image segment; double to 40 and 80
  when the barrier changes by more than 0.05 kcal mol\(^{-1}\).
- The diagnostic evaluator is separate and never enters the optimization
  \(N_F\) axis.

## Why the apparent two-well alpha curve occurs

It is not a plotting artifact.  Two mechanisms are present.

1. **Discrete inner-solve/tangent-refresh transition.**  With cap 20 and
   KKT tolerance 0.8, alpha 1.5 typically completes a frozen-tangent
   subproblem after only about 20 total force evaluations.  At alpha 1.75 the
   median jumps to about 400, and the fraction of completed subproblems with
   at least one image hitting the cap jumps from about 0.11 to 0.64.  Thus
   nearby alphas receive very different numbers of outer tangent refreshes at
   the same total-force budget.

2. **True nonlinear transient/path branching.**  A matched KKT=0.7, cap-30
   grid preserves alpha-1 performance but still gives, at \(N_F\le 2000\),
   endpoint \(I_\perp\) values
   \(0.142,0.836,1.709,2.103,3.286,1.164,0.427,0.867\) for
   alpha \(1,1.25,1.5,1.75,2,2.5,3,10\), respectively.  The final-path RMS
   coordinate difference jumps to 0.067 Å between alpha 1 and 1.25.
   Tightening one common KKT threshold therefore moves but does not eliminate
   the secondary valleys.

A single-inner-step control largely suppresses the good large-alpha basin,
but more than 90% of high-alpha subproblems hit the one-step cap.  At
\(N_F\approx2000\), its final \(I_\perp\) values for alpha
\(1,1.25,1.5,1.75,2,3,5,10\) are approximately
\(0.137,0.506,0.967,0.958,3.578,0.917,1.979,4.043\).
This is useful causal evidence, but it is an intentionally very inexact IVM
variant and is not acceptable as the theoretical production experiment.

The honest conclusion is that a globally smooth single well cannot be
obtained here while simultaneously retaining the production algorithm,
using one alpha-independent subproblem rule, and plotting raw finite-budget
endpoints.  Producing one would require an alpha-dependent accuracy/cost rule,
regression/smoothing, endpoint selection, or switching to the intentionally
inexact one-step variant.  Those choices would confound the alpha study.

## Cap ranking and paper consequence

The preregistered auxiliary score ranks the caps:

| cap | aggregate score | rank |
|---:|---:|---:|
| 20 | -0.4760491750 | 1 |
| 18 | -0.5534613008 | 2 |
| 15 | -0.7145309251 | 3 |

Higher is better, but all three fail the stronger requested visual criterion:
their raw \(I_\perp\) curves remain rugged, and endpoint/tail consistency
fails at most later budgets.  This score is therefore diagnostic, not a
license to replace the production method.

The existing alpha-1 production result remains substantially better than the
best comparison methods and does not need to be redrawn:

- IVM: final \(N_F=6823\), \(I_\perp=0.0388524\), densified barrier
  20.3737 kcal mol\(^{-1}\).
- ODE-NEB: final \(N_F=20002\), \(I_\perp=0.184711\), densified barrier
  23.7579 kcal mol\(^{-1}\).
- OBSM-FIRE: final \(N_F=19982\), \(I_\perp=0.212422\), densified barrier
  24.1681 kcal mol\(^{-1}\).

Because cap 18/20 reproduce the alpha-1 production path rather than improve
it, the current article comparison figures remain the correct production
figures.

## Primary artifacts

- Complete alpha/cap figure:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/figures/reviewer_subproblem_cap_alpha_well_20260729a/chignolin_ivm_subproblem_cap_alpha_sensitivity_total_nf.pdf`
- 57-run cap comparison:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/figures/reviewer_subproblem_cap_comparison_20260729a/`
- Strict cap-15 dense manifest:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/reports/reviewer_subproblem_cap15_dense_strict_20260729a/manifest.json`
- Strict cap-18/20 dense manifest:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/reports/reviewer_subproblem_caps18_20_dense_20260729a/manifest.json`
- Final cap-18/20 optimization manifests:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/reports/finalized_subproblem_caps18_20_20260729a/manifest_index.json`
- Preregistered score:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/reports/reviewer_subproblem_cap_score_20260729a/subproblem_cap_well_score_formula.json`
- KKT=0.7 mechanism control:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/figures/reviewer_alpha_double_well_kkt07_20260729a/`
- One-step mechanism control:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/figures/reviewer_alpha_double_well_cap1_control_20260729a/`

All relevant automated semantic tests pass, including total-force accounting,
missing-budget gaps, combined-manifest cap filtering, and no smoothing or
carry-forward in the final plot.
