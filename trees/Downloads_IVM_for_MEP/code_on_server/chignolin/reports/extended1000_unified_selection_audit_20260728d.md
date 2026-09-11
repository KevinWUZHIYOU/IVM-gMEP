# Chignolin 1000-call extension, unified selection, and figure audit

Date: 2026-07-28

## Scope

This report audits the final Chignolin comparison bundle in
`figures/article_update_20260728d`.  Earlier result and figure directories were
left unchanged.

The final comparison uses:

- the preserved fixed-alpha IVM production run;
- newly selected and independently confirmed 1000-call-budget runs for
  ODE-NEB, Static-NEB, Original String, and OBSM-FIRE;
- total joint energy-force evaluations over all 22 images as the cost axis;
- an independent diagnostic evaluator for densified barriers, whose calls are
  never included in the optimization cost.

All production methods start from the same path with SHA-256
`57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996`.

## Predeclared shared selection score

Every candidate, irrespective of method family, was scored by the same
absolute formula (lower is better):

```text
0.30*AUC(log10(I_perp))
+ 0.25*AUC(log10(densified barrier))
+ 0.20*log10(final I_perp)
+ 0.15*log10(final densified barrier)
+ 0.10*log10(1 + final equidistance integral)
```

There is no within-family rank normalization and no method-specific score.
Sparse barriers were used only for screening.  The final selection replaces
them with adaptive densified barriers.

The search and validation stages comprised:

- 83 ultrafine 300-call screening runs: 17 Original String `dt` values and
  66 Static-NEB `(k, alpha)` combinations;
- six independent 500-call intermediate productions, the top three from each
  of those two families;
- 34 independent 1000-call-budget candidates: three Original String, three
  Static-NEB, eight ODE-NEB, and 20 focused OBSM-FIRE configurations;
- adaptive densified-barrier validation of the top three candidates from each
  of the four non-IVM families (12 runs);
- four independent confirmation productions, one for each selected non-IVM
  method, followed by independent dense-barrier postprocessing.

The four confirmation productions and dense results are bitwise identical to
the selected candidate results for the audited arrays; no tuning trajectory
was spliced into a final curve.

## Selected fixed configurations

| Method | Selected configuration | Shared dense score |
|---|---|---:|
| ODE-NEB | `k = 50` (ODE; the alpha interface is inactive) | 0.652320 |
| OBSM-FIRE | `lambda = 0.9`, 5 inner steps, FIRE `dt = 0.12`, `beta = 1`, reparameterize every outer step | 0.664366 |
| Original String | full unprojected physical-force descent, `dt = 0.012575`, cubic-spline reparameterization every step | 0.767058 |
| Static-NEB | `k = 40`, `alpha = 0.012575` | 1.026789 |

Original String and Static-NEB are no longer treated as indistinguishable.
Under the same score, Original String is the better overall fixed
configuration.  Static-NEB has a slightly lower terminal `I_perp`
(0.701274 versus 0.703575), while Original String has a lower terminal dense
barrier (26.500720 versus 26.594264 kcal/mol) and a much smaller equidistance
integral (1.22528 versus 879.929).

Original String uses the complete physical force for every image update and
does not project that force.  Its reparameterization is the Appendix-A cubic
spline used by the Simplified String Method.  All shared path diagnostics use
the common normalized central-difference tangent.

## Final production endpoints

| Method | Total force evaluations | Max calls/internal image | `I_perp` | Dense barrier (kcal/mol) | Physical `fmax` (eV/A) | Equidistance integral |
|---|---:|---:|---:|---:|---:|---:|
| IVM | 6,823 | 342 | 0.0388524 | 20.373740 | 0.00806290 | 5.89e-20 |
| ODE-NEB | 20,002 | 1,000 | 0.184711 | 23.757942 | 0.0211225 | 183.684 |
| Static-NEB | 20,002 | 1,000 | 0.701274 | 26.594264 | 0.0300572 | 879.929 |
| Original String | 20,002 | 1,000 | 0.703575 | 26.500720 | 0.0300972 | 1.22528 |
| OBSM-FIRE | 19,982 | 999 | 0.212422 | 24.168121 | 0.0159613 | 159.099 |

OBSM-FIRE is run with a 1000-call hard upper budget.  Its indivisible outer
cycle contains a fixed batch of evaluations; the final legal state is
999 calls/internal image because the next complete cycle would exceed the
budget.  The 19,982 total is therefore the audited algorithmic cost, not a
missing or rounded count.

At the common precision thresholds used for the cost-to-precision arrows:

- `I_perp = 0.212422`: IVM needs 1,402 evaluations, versus 17,662 for ODE-NEB
  and 19,942 for OBSM-FIRE, giving 12.6x and 14.2x speedups;
- dense barrier = 24.168121 kcal/mol: IVM needs 1,502 evaluations, versus
  16,562 for ODE-NEB and 19,982 for OBSM-FIRE, giving 11.0x and 13.3x speedups.

## Fixed-alpha IVM extension

The sensitivity set now contains 11 real fixed-alpha productions:

```text
0.03, 0.1, 0.3, 1, 3, 10, 20, 30, 100, 300, 1000
```

The last four are newly computed experiments.  In every saved trajectory,
`alpha_mode` is `fixed`, every recorded `active_alpha` equals the nominal
value, and `alpha_max` does not clip it.  The horizontal alpha range therefore
reaches `10^3` using measured results rather than an empty axis extension.

The three plotted total-force checkpoints are exactly 1,002, 3,002, and 6,002
evaluations for every alpha.  Dense diagnostic costs are recorded separately.

## Counter and barrier audits

- Each production has 22 independent image counters.
- Every final counter vector equals the vector in its audit JSON.
- Counter vectors are monotone.
- For every production, the raw backend joint-evaluation count equals the sum
  of all image joint-evaluation counts.
- Repeated force/energy requests at a cached coordinate do not double count.
- Every adaptive dense-barrier result has
  `diagnostic_backend_raw_joint_evaluations ==
  diagnostic_evaluator_cache_misses`.
- Every dense audit marks its calls as excluded from the optimization axis.
- Adaptive subdivision counts for the four confirmed baselines were:
  OBSM-FIRE `(20:48, 40:0, 80:0)`, ODE-NEB `(20:45, 40:2, 80:1)`,
  Original String `(20:42, 40:5, 80:1)`, and
  Static-NEB `(20:42, 40:5, 80:1)`.

The primary cost field in the figure summary is
`counter_vectors.sum(axis=1)`, labeled `Number of Force Evaluations`.

## Software and artifact validation

- 53/53 dependency-free tests passed, including cache-history reuse within a
  run, no cache sharing between experiments, force/energy request identity,
  rejected-trial accounting, checkpoint restoration, central-difference
  tangent consistency, and unprojected Original String descent.
- The three article PDFs are one-page vector files with zero embedded raster
  images and zero unembedded fonts.
- Their SVG counterparts contain no `<image>` or `data:image` elements.
- Exported CSV data contain no `NaN` or infinite numeric values.

An early dense-postprocessing launch was stopped after it read an unfiltered
candidate manifest.  Its partial/unselected outputs are absent from every
final manifest and figure.  The final dense runner takes an explicit production
manifest, and the final figures reference only the four independent
confirmation runs plus the preserved IVM production.

## Final artifacts

- `figures/article_update_20260728d/chignolin_main_metrics_total_nf.pdf`
- `figures/article_update_20260728d/chignolin_cost_to_precision_total_nf.pdf`
- `figures/article_update_20260728d/chignolin_ivm_fixed_alpha_sensitivity_total_nf.pdf`
- `figures/article_update_20260728d/chignolin_method_comparison_summary.json`
- `reports/confirmed1000_productions_20260728c/manifest.json`
- `reports/confirmed1000_dense_20260728c/manifest.json`
- `reports/extended1000_dense_scoring_20260728b/manifest.json`
- `reports/ivm_fixed_alpha_extended_20260728a/manifest.json`
- `reports/ivm_fixed_alpha_extended_20260728a/dense_manifest.json`

The read-only audit helper used for this report is
`working/chignolin_new_methods/audit_article_update_20260728d.py` in the local
project mirror.
