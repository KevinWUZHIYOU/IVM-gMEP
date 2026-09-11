# Chignolin V9 production notes

## Locked experimental facts

- The common 22-image path is ordered **folded to misfolded**:
  `initial_path[0]` exactly equals `folded_min.traj`, and
  `initial_path[-1]` exactly equals `misfolded_min.traj`.
- Every method uses the identical initial-path SHA-256
  `57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996`.
- All optimization and diagnostic tangents use the normalized central
  difference `x[i+1] - x[i-1]`.
- The production V9 IVM fixes `alpha=1`; it has no alpha schedule and no
  adaptive alpha controller.
- The same alpha-independent subproblem rule is used for every sensitivity
  member: projected L-BFGS, history 7 within the frozen subproblem, relative
  KKT forcing factor 0.7, at most 30 new joint energy-force evaluations per
  image, exact proximal-objective backtracking, and a geometry-scaled trust
  radius with a common safety cap.
- Curvature pairs are accumulated inside each frozen-normal subproblem.
  Cross-subproblem history is intentionally disabled because the
  central-difference normal spaces rotate between outer IVM steps.
- Parameterization uses identical nearest-neighbour spring flow with the same
  central tangent. It uses no cubic spline and no OpenMM energy or force.
- A projected whole-band trial is evaluated only after its call-free spring
  projection. After state-defined projected stalling, the algorithm continues
  the same fixed-alpha frozen-normal proximal subproblem and applies the
  smallest call-free spring correction needed to restore a relative
  squared-speed defect of `2.5e-7` when it exceeds `5e-7`.
- There is no outer refreshed-`I_perp` acceptance guard in the selected
  `proximal_only` run. Rejected inner trials remain charged.
- The cost axis is `counter_vectors.sum(axis=1)`, i.e. all real joint
  energy-force cache misses over all 22 images. Diagnostic densification is
  audited separately and excluded.

## Production alpha=1 values at the latest complete macrostate

- `N_F = 9969`, maximum internal-image call label 505.
- Unnormalized `I_perp = 0.029596057873027555`.
- Physical normal `fmax = 0.007064525631970911 eV/Angstrom`.
- Raw equidistance integral `I_eq = 207.10404238537322`.
- Relative equidistance defect `2.5e-7`.
- Segment-length coefficient of variation `1.7501259576779382e-4`.
- Every image satisfies the relative-KKT target at this macrostate:
  maximum relative KKT 0.5056679215672303, zero unconverged and zero
  call-limited images.
- Sparse 22-image barrier is `19.359388024621275 kcal/mol` and is
  monitoring-only.
- Real OpenMM adaptive densification gives
  `20.681000463452797 kcal/mol`; 10 and 20 subdivisions agree at this path,
  with 421 diagnostic evaluations, all excluded from `N_F`.

## Comparator endpoints

- ODE-NEB: `N_F=20002`, `I_perp=0.18471115741866675`,
  dense barrier `23.75794213237218 kcal/mol`,
  `I_eq=183.6842721377893`.
- OBSM-FIRE: `N_F=19982`, `I_perp=0.21242215133173672`,
  dense barrier `24.168121352641172 kcal/mol`,
  `I_eq=159.09873201689604`.
- Static-NEB: `N_F=20002`, `I_perp=0.7012736730010077`,
  dense barrier `26.59426369566459 kcal/mol`.
- Original String: `N_F=20002`, `I_perp=0.7035750340641046`,
  dense barrier `26.50072035457217 kcal/mol`.

## Dense-barrier protocol

For every scored or plotted path, linearly interpolate each of the 21
Cartesian image segments and evaluate real OpenMM energies at 10 and 20
subdivisions per segment. If successive barrier estimates differ by more than
`0.05 kcal/mol`, double to 40 and then 80. The barrier is the maximum dense
energy minus the folded (`path[0]`) endpoint energy. Sparse barriers never
enter scores, cost-to-precision curves, figures, or manuscript claims.

## Final alpha selection and measured well

- The complete measured grid is
  `alpha = 0.3, 0.5, 0.75, 1, 1.25, 1.5, 2`.
- Every member uses the same KKT factor `0.7`, inner cap `30`, line search,
  trust rule, spring projection, stopping rule, initial band, and
  `N_F <= 10,000` analysis budgets.  Only alpha changes.
- The primary stationarity statistic is the force-axis arithmetic mean over
  the final 30% of each requested budget.  It has a unique measured minimum
  at alpha=1 for `N_F <= 1,000, 3,000, 6,000, 10,000`.
- The primary barrier statistic is the RMS error of seven adaptively
  densified real-OpenMM barriers in the same final window relative to the
  independently frozen prior cap=15 reference
  `20.373739515742795 kcal/mol`.  Its unique measured moderate-alpha minimum
  is alpha=1 for the first three budgets and alpha=0.75 at `N_F <= 10,000`.
- The final-window stationarity values at `N_F <= 10,000`, in alpha order,
  are
  `[0.0682915, 0.0508465, 0.0401189, 0.0312567, 0.0532028, 0.352012, 0.258389]`.
- The corresponding final-window dense-barrier RMS errors are
  `[1.49524, 0.402023, 0.0772776, 0.209299, 0.664604, 3.85414, 3.45568]`
  kcal/mol.
- Alpha=1 is selected by the predeclared lexicographic score and passes every
  production gate.  Small measured large-alpha endpoint shoulders are
  retained and are not smoothed, fitted, or deleted.
- Authoritative selection report:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/reports/v9_kkt0p7_cap30_final_selection_20260730/V9_ALPHA_FINAL_WINDOW_SCORES.json`.

## Final article figures and hashes

- Main metrics:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/figures/article_update_20260730_v9_final/chignolin_main_metrics_total_nf.pdf`
  (`b808143a8046080c9258cd4ccef37414d5153be8ad4d30a9cfdc508e3dd99887`).
- Cost to precision:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/figures/article_update_20260730_v9_final/chignolin_cost_to_precision_total_nf.pdf`
  (`c56e61e7575b7f1cc784b946bd37e65682204579f7f85c3115b200b2dbdf37ff`).
- Fixed-alpha sensitivity:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/figures/article_update_20260730_v9_final/chignolin_ivm_fixed_alpha_sensitivity_total_nf.pdf`
  (`6ac9c94495d0c7ce77d3f4fb99356f70e6a3264cd1c64f9b4be1eb3fc0a39c4d`).
- Updated article source:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/main.tex`
  (`50b6feb8f3beebd45597343bae7dc1fc45a90ca104db66c6836743f7fbb3acbd`).
- Compiled article:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/main.pdf`
  (`43fceb829d5da9e67c77ad86f3cf33df9d4ec61f0db8bf0565cb7b9ecbd0e765`).

## First-passage values from the production comparison

- At the strictest common `I_perp` threshold of `0.21242215133173672`,
  IVM, ODE-NEB, and OBSM-FIRE require `1382`, `17662`, and `19942`
  total force evaluations, respectively: `12.7800x` versus ODE-NEB and
  `14.4298x` versus OBSM-FIRE.
- At the strictest common adaptive dense-barrier threshold of
  `24.168121352641172 kcal/mol`, IVM, ODE-NEB, and OBSM-FIRE require
  `1502`, `16562`, and `19982` total force evaluations, respectively:
  `11.0266x` versus ODE-NEB and `13.3036x` versus OBSM-FIRE.
