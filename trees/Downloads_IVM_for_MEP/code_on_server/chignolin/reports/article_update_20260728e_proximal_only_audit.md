# Chignolin article figures with proximal-only outer acceptance

Date: 2026-07-28

## Scope

The following three article figures were regenerated with the selected
Chignolin IVM run using fixed \(\alpha=1\), exact proximal-objective line
search, and `state_globalization=proximal_only`. The refreshed outer
\(I_\perp\) acceptance guard is disabled.

The previous `article_update_20260728d` directory and all protected production
results remain unchanged. New outputs are stored in
`figures/article_update_20260728e`.

## Selected alpha=1 trajectory

The full proximal-only \(\alpha=1\) run stops at call 342 for the same
L-BFGS-B no-progress condition as the protected production run. With NaNs
treated as equal, every common stored field is identical except:

- configuration hash;
- run label;
- wall-clock timestamps.

In particular, all paths, counter vectors, \(I_\perp\), sparse barriers,
\(f_{\max}\), accepted fractions, and the final path are identical. Therefore
removing the outer \(I_\perp\) guard does not alter the selected trajectory or
its reported efficiency.

The regenerated main-metrics and cost-to-precision CSV files are byte-for-byte
identical to the `20260728d` files. Their 600-dpi PNG files are also
byte-for-byte identical. Their summaries now point to the proximal-only
result and explicitly record:

```text
alpha = 1
state_globalization = proximal_only
outer_i_perp_acceptance_guard = false
proximal_objective_line_search = true
```

The displayed same-precision speedups remain:

| metric | vs ODE-NEB | vs OBSM-FIRE |
|:---|---:|---:|
| unnormalized \(I_\perp\) | 12.6x | 14.2x |
| densified 0 K barrier | 11.0x | 13.3x |

## Proximal-only fixed-alpha sweep

Eleven fixed values were run with the same initial path, central-difference
tangent, proximal subproblem solver, trust settings, and no outer
\(I_\perp\) guard:

\[
\alpha\in\{0.03,0.1,0.3,1,3,10,20,30,100,300,1000\}.
\]

The figure uses total-force budget caps. At each cap it plots the last
measured state at or below that cap. The \(\alpha=100\) run reached a
certified no-progress terminal state at call 242, corresponding to
\(N_F=4842\); this terminal state is carried forward for the
\(N_F\le6002\) curve without inventing additional force calls.

### Unnormalized I_perp

| \(\alpha\) | \(N_F\le1002\) | \(N_F\le3002\) | \(N_F\le6002\) |
|---:|---:|---:|---:|
| 0.03 | 34.5824 | 3.42457 | 1.16216 |
| 0.1 | 3.33099 | 0.636164 | 0.256479 |
| 0.3 | 0.827834 | 0.198087 | 0.100371 |
| 1 | 0.323773 | 0.0968068 | **0.0460167** |
| 3 | 0.466349 | 0.653386 | 0.117192 |
| 10 | 0.325116 | **0.0914670** | 0.0497513 |
| 20 | 1.25901 | 0.457945 | 0.121324 |
| 30 | **0.210346** | 1.44820 | 0.0794964 |
| 100 | 6.96955 | 0.328069 | 0.340578 |
| 300 | 10.4225 | 0.343228 | 2.29086 |
| 1000 | 6.81051 | 8.99940 | 0.174680 |

The no-guard right side is not a smooth explicit-step divergence curve.
Instead it is strongly nonmonotone: for example, \(\alpha=1000\) has
\(I_\perp=8.9994\) at \(N_F=3002\), while \(\alpha=300\) rebounds to
\(I_\perp=2.2909\) at \(N_F=6002\). This is the expected observable signature
of difficult, weakly regularized nonconvex proximal subproblems. The selected
\(\alpha=1\) is the best late-budget value; \(\alpha=10\) is slightly better
at the middle budget and \(\alpha=30\) at the earliest budget, which is
reported without suppression.

### Densified 0 K barrier

| \(\alpha\) | \(N_F\le1002\) | \(N_F\le3002\) | \(N_F\le6002\) |
|---:|---:|---:|---:|
| 0.03 | 48.1430 | 31.7834 | 28.2004 |
| 0.1 | 31.7254 | 26.3182 | 24.2601 |
| 0.3 | 27.1513 | 23.8531 | 22.7967 |
| 1 | 24.7706 | 22.7272 | 20.3567 |
| 3 | 25.7485 | 23.7432 | 22.8712 |
| 10 | 24.0740 | 21.4828 | 20.4687 |
| 20 | 22.7625 | 20.4688 | 20.7004 |
| 30 | 21.7816 | 20.7031 | 20.6871 |
| 100 | 21.8622 | 20.6992 | 20.7067 |
| 300 | 22.3256 | 20.6244 | 20.7956 |
| 1000 | 22.2627 | 20.3921 | 20.7157 |

The large-alpha barriers remain near 20--21 kcal/mol even when
\(I_\perp\) is poor. This confirms that a transient low barrier is not by
itself evidence of MEP convergence.

## Accounting and validation

- Every optimization run uses `counter_vectors.sum(axis=1)` as the plotted
  Number of Force Evaluations.
- The 11 fixed-alpha runs contain 64,687 audited OpenMM joint
  energy-force evaluations in total.
- Independent barrier diagnostics contain 13,849 audited OpenMM evaluations.
  They are excluded from every optimization cost axis.
- Every fixed-alpha backend count equals the sum of the 22 image evaluators.
- Every diagnostic backend count equals its diagnostic cache misses.
- All active-alpha arrays contain only their declared fixed value.
- All runs use the common initial-path SHA-256
  `57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996`.
- All production and method tests pass: 53/53.
- The three requested PDFs are one-page vector files, contain no raster
  images, and have all fonts embedded.
- Rendered PDF pages were visually checked for clipping, overlap, and
  legibility.

## Final files

- `figures/article_update_20260728e/chignolin_main_metrics_total_nf.pdf`
- `figures/article_update_20260728e/chignolin_cost_to_precision_total_nf.pdf`
- `figures/article_update_20260728e/chignolin_ivm_fixed_alpha_sensitivity_total_nf.pdf`
- `reports/ivm_fixed_alpha_proximal_only_20260728a/manifest.json`
- `reports/ivm_fixed_alpha_proximal_only_20260728a/dense_manifest.json`

