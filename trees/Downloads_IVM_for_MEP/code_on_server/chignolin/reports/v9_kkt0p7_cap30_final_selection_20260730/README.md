# Chignolin V9 fixed-alpha production result

## Selected method

The selected trajectory is one unspliced fixed-alpha IVM run:

- alpha: `1`
- tangent: normalized central difference at every optimization and diagnostic
  use
- frozen-normal subproblem solver: projected L-BFGS
- L-BFGS history: `7` curvature pairs, retained only inside the current
  frozen subproblem
- KKT stopping target:
  `max(1e-8, 0.7 * initial_subproblem_KKT)`
- inner limit: at most `30` new joint energy-force evaluations per image
- globalization: exact proximal-objective Armijo backtracking
- outer refreshed-I_perp acceptance guard: disabled
- parameterization: geometry-only identical nearest-neighbour spring flow
  using the central tangent
- spline use in IVM: none
- OpenMM calls made by the spring projection: `0`
- late spring correction: trigger relative squared-speed defect `5e-7`,
  target `2.5e-7`

The algorithm never requests an OpenMM Hessian and never estimates the full
potential Hessian by finite differences.  L-BFGS supplies quasi-Newton
curvature from the current subproblem, and the molecular bonded-block
preconditioner is constructed analytically without OpenMM queries.

## Common experiment and cost

All methods use the same saved 22-image IDPP path, ordered folded to
misfolded, with SHA-256
`57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996`.
There are 20 moving images and two fixed endpoints.  OpenMM uses the CPU
platform in double precision with `amber14-all.xml`,
`implicit/gbn2.xml`, and `NoCutoff`.

The primary cost is
`N_F = counter_vectors.sum(axis=1)`: all real joint energy-force cache
misses over all 22 images.  Each image has an independent run-local
exact-coordinate cache.  Energy and force are always evaluated and cached
together.  Repeated API requests and exact returns to a coordinate already
evaluated in the same run are cache hits; rejected and rolled-back evaluated
trials remain charged.  Caches are never shared between experiments.

## Dense 0 K barrier

Sparse 22-image barriers are monitoring-only.  Every barrier used for
selection, scoring, cost-to-precision, or figures is recomputed with real
OpenMM energies on a densified Cartesian path:

1. divide each of the 21 adjacent image segments into 10 and 20
   subsegments;
2. compare successive barrier estimates;
3. if the change exceeds `0.05 kcal/mol`, double to 40 and, if needed, 80
   subsegments per segment;
4. subtract the folded endpoint energy from the largest densified energy.

Diagnostic evaluations use a separate audited evaluator and are excluded
from `N_F`.

## Production endpoint and baselines

At its latest complete macrostate before `N_F=10,000`, V9 alpha=1 has:

- `N_F = 9969`
- unnormalized `I_perp = 0.029596057873027555`
- physical normal `fmax = 0.007064525631970911 eV/Angstrom`
- adaptive dense barrier `20.6810004634525 kcal/mol`
- equidistance integral `207.10404238537322`
- relative squared-speed defect `2.5e-7`
- segment-length CV `1.7501259576779382e-4`
- maximum relative KKT `0.5056679215672303`
- zero unconverged or call-limited images

Comparator endpoints:

| Method | Final `N_F` | `I_perp` | Dense barrier (kcal/mol) | Equidistance integral |
|---|---:|---:|---:|---:|
| IVM | 9969 | 0.0295961 | 20.6810 | 207.104 |
| ODE-NEB | 20002 | 0.184711 | 23.7579 | 183.684 |
| OBSM-FIRE | 19982 | 0.212422 | 24.1681 | 159.099 |
| Static-NEB | 20002 | 0.701274 | 26.5943 | 879.929 |
| Original String | 20002 | 0.703575 | 26.5007 | 1.22528 |

At the strictest common `I_perp=0.212422` threshold, IVM uses 1382 total
force evaluations versus 17662 for ODE-NEB and 19942 for OBSM-FIRE:
`12.78x` and `14.43x` speed-ups.  At the common adaptive dense-barrier
threshold `24.1681 kcal/mol`, the respective costs are 1502, 16562, and
19982: `11.03x` and `13.30x`.

Original String uses the true unprojected physical force followed by the
Simplified String Method cubic-spline reparameterization.  OBSM uses the
same cubic-spline implementation.  IVM does not.

## Fixed-alpha selection

The measured grid is `0.3, 0.5, 0.75, 1, 1.25, 1.5, 2`.  Every member uses
the same KKT factor, inner cap, trust and line-search rules, spring
projection, stopping rule, initial path, and force budgets; only alpha
changes.

The stationarity ordinate is the force-axis arithmetic mean over the final
30% of each budget `N_F <= 1,000, 3,000, 6,000, 10,000`.  It has a unique
measured minimum at alpha=1 at all four budgets.  The barrier ordinate is the
RMS error of seven adaptively densified real-OpenMM barriers in the same
final window relative to the independently frozen prior cap=15 reference
`20.373739515742795 kcal/mol`.  Its unique moderate-alpha minimum is alpha=1
for the first three budgets and alpha=0.75 at the largest budget.  All
measured points, including small large-alpha endpoint shoulders, are retained
and joined directly; no regression, smoothing, or deletion is used.

The predeclared lexicographic score and production gates select alpha=1.
Tighter-KKT/cap=60 and cap=60/KKT=0.7 controls are archived as negative
controls because they degrade alpha=1 or fail to remove the measured late
shoulder.

## Authoritative outputs

- Production run:
  `runs/ivm/ivm_fixed_alpha_v9_kkt0p7_cap30_a1_central_state_ivm_cfg9b5a238786a3_budget650.npz`
- Production audit:
  same basename with `.audit.json`
- Full alpha summary:
  `reports/v9_kkt0p7_cap30_final_grid_20260730/v9_kkt0p7_cap30_grid_summary.json`
- Dense diagnostic manifest:
  `reports/v9_kkt0p7_cap30_final_dense_20260730/dense_manifest.json`
- Final selection:
  `reports/v9_kkt0p7_cap30_final_selection_20260730/V9_ALPHA_FINAL_WINDOW_SCORES.json`
- Final figures:
  `figures/article_update_20260730_v9_final/`
- Analysis contract:
  `src/V9_FINAL_WINDOW_ANALYSIS_CONTRACT_20260730.md`

## Reproduction

Run or resume one fixed-alpha member:

```sh
zsh /Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/src/launch_v9_kkt0p7_cap30_grid_member_20260730.zsh 1 650
```

Regenerate summaries, dense diagnostics, selection, and figures after all
grid members exist:

```sh
zsh /Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/src/finalize_v9_production_20260730.zsh
```

The finalizer uses the lowercase portable grid-summary filename
`v9_kkt0p7_cap30_grid_summary.json`.

## Verification

- all 63 external Chignolin regression tests pass, including 47 core
  cache/accounting and solver tests, five string-method tests, two V2
  line-search tests, two transactional-relock tests, four V8 policy tests,
  and three V9 spline-free spring-projection tests;
- backend OpenMM calls equal the summed per-image cache misses;
- dense diagnostic backend calls equal diagnostic cache misses and are
  excluded from optimization cost;
- the selected curve comes from one production file;
- the preserved cap=15/18/20/25/30 manifest still matches all 31 original
  files.
