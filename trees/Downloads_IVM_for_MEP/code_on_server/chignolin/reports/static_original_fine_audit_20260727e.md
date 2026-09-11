# Chignolin Original String vs Static-NEB fine experiment audit

Date: 2026-07-27

## Outcome

The finer experiment resolves a real but small performance difference. Using
the prespecified whole-curve rank weights, Original String is the overall
winner:

| method | weighted rank score (lower is better) |
|---|---:|
| Original String | 0.45 |
| Static-NEB | 0.55 |

This result is not a claim that Original String dominates every metric.
Static-NEB has the lower terminal `I_perp` and the lower densified-barrier
whole-curve AUC. Original String has the lower `I_perp` whole-curve AUC, lower
terminal densified barrier, and much better parameterization quality.

## Method definitions

- Original String uses a full, unprojected physical-force step for every
  internal image:
  `x_i <- x_i + dt * F_i`.
- No tangent projection is used in the Original String descent update.
- After every Original String descent step, the path is redistributed by the
  Simplified String Method Appendix A cubic-spline arclength
  reparameterization.
- Static-NEB uses the shared normalized central-difference tangent and the
  standard spring-length-difference force.
- Both methods start from the same path with SHA-256
  `57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996`.
- The primary cost axis is
  `counter_vectors.sum(axis=1)`, labeled **Number of Force Evaluations**.

## Fine selection

The old refinement ended at `0.0125`. The new boundary experiment explicitly
resolved the untested interval between the stable `0.0125` runs and the
previously observed unstable point near `0.01382`.

| stage | method | configurations | budget per internal image |
|---|---|---:|---:|
| boundary | Original String | 15 | 60 |
| boundary | Static-NEB | 119 | 60 |
| fine | Original String | 5 | 300 |
| fine | Static-NEB | 24 | 300 |

The boundary grid used a `1e-4` increment. Every screening trajectory was
independent and was used only for parameter selection; no screening trajectory
was spliced into the paper curves.

Selected fixed configurations:

- Original String: `dt = 0.0125`, reparameterization interval `1`.
- Static-NEB: `k = 40`, `alpha = 0.0126`.

Original String at `dt = 0.0126` had a slightly better 300-call endpoint but a
worse whole-curve `I_perp` AUC because of its early oscillation. It was
therefore not selected by endpoint cherry-picking.

## Independent 500-call production results

Each selected configuration was rerun independently from the common initial
path and saved at all optimization checkpoints.

| metric | Original String | Static-NEB | better |
|---|---:|---:|---|
| total Number of Force Evaluations | 10,002 | 10,002 | tied |
| `I_perp` log-AUC | 1.0726824 | 1.0738295 | Original |
| terminal `I_perp` | 1.8699550 | 1.8454158 | Static |
| densified barrier log-AUC | 1.6284463 | 1.6273893 | Static |
| terminal densified barrier (kcal/mol) | 29.7893494 | 29.8001454 | Original |
| terminal equidistance integral | 0.5250853 | 404.9708284 | Original |
| terminal segment-length CV | 2.10597e-6 | 6.24180e-4 | Original |

The `I_perp` curves have one persistent crossover. Original String is lower
through 2,242 total force evaluations; Static-NEB is lower from 2,262 total
force evaluations onward.

The densified barrier is lower for Static-NEB at most intermediate
checkpoints, but the final persistent ordering changes near 7,842 total force
evaluations. At 10,002 total force evaluations, Original String is lower by
0.0107960 kcal/mol.

## Prespecified composite decision

The two-method comparison uses the same whole-curve priorities used by the
screening design, with the publication-grade densified barrier replacing the
sparse screening barrier:

| component | weight | winner |
|---|---:|---|
| `I_perp` log-AUC | 0.30 | Original String |
| densified barrier log-AUC | 0.25 | Static-NEB |
| terminal `I_perp` | 0.20 | Static-NEB |
| terminal densified barrier | 0.15 | Original String |
| terminal equidistance integral | 0.10 | Original String |

With two methods, the winner of a component receives rank fraction 0 and the
loser receives rank fraction 1. This gives Original String 0.45 and Static-NEB
0.55. The difference is therefore resolved without changing a weight after
looking at the result.

## Dense barrier and counter audit

- 48 optimization checkpoints were evaluated for each production path.
- Adaptive subdivisions per segment were `20` for 42 paths, `40` for 5 paths,
  and `80` for 1 path, identically for both methods.
- Each dense diagnostic used 23,476 joint energy-force OpenMM evaluations.
- Diagnostic calls were recorded separately and excluded from the optimization
  cost axis.
- For each production:
  `backend_raw_joint_evaluations = sum_image_joint_evaluations = 10,002`.
- For each dense diagnostic:
  `diagnostic_backend_raw_joint_evaluations =
  diagnostic_evaluator_cache_misses = 23,476`.
- All 163 boundary/fine screening records have a consistent counter audit and
  the common initial-path hash.

## Validation

- 53 dependency-free unit tests passed.
- The tests include the explicit assertion that the Original String update
  uses the full unprojected force.
- All four article candidate PDFs are one-page vector figures with embedded
  fonts.
- PDF pages were rendered to PNG and visually checked for clipping, overlap,
  and label legibility.

## Output

- Article candidate directory:
  `figures/article_update_20260727e`
- Fine production manifest:
  `reports/paper_static_original_fine_20260727e/manifest.json`
- Dense diagnostic manifest:
  `reports/paper_static_original_fine_dense_20260727e/manifest.json`
- Machine-readable comparison summary:
  `figures/article_update_20260727e/chignolin_method_comparison_summary.json`
- Dedicated resolution figure:
  `figures/article_update_20260727e/chignolin_static_neb_vs_original_string_resolved_total_nf.pdf`
