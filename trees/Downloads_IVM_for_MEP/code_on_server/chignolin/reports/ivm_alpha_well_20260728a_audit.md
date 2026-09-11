# Chignolin fixed-alpha well and large-alpha audit

## Outcome

Two new fixed-alpha sensitivity figures were generated without overwriting the
existing article figures:

- `chignolin_ivm_fixed_alpha_well_logy`
- `chignolin_ivm_fixed_alpha_well_lineary`

They use identical raw points and an identical fitted guide; only the vertical
axis transformation differs.

The focus range is
\(\alpha\in[0.03,100]\). This is the range in which the proximal term remains
numerically distinguishable from the common trust cap. Separate
\(\alpha=10^4\) and \(10^5\) probes are retained as saturation diagnostics and
are not used to force the main trend into an artificial high-alpha divergence.

## Figure definition

All runs use the same equidistant initial path, centered-difference tangent,
fixed alpha, `state_globalization=proximal_only`, exact proximal-objective line
search, and no outer \(I_\perp\) acceptance guard.

The horizontal coordinate is fixed \(\alpha\). Both panels use the largest
common total-force budget
\[
N_F\le 6002,
\]
where \(N_F\) is the sum of joint energy-force cache misses over all 22 images.

- Panel a plots the minimum unnormalized \(I_\perp\) actually attained by that
  budget.
- Panel b plots the lowest independently densified raw 0 K forward barrier
  measured by that budget.

The hollow circles are all 17 independent fixed-alpha observations in
\([0.03,100]\). The solid curve is only a guide to the broad well: a quadratic
in \(\log_{10}\alpha\), fitted by Huber iteratively reweighted least squares.
For panel a the response is \(\log_{10} I_\perp\); for panel b it is the raw
barrier in kcal/mol. The raw observations remain visible, and the CSV contains
all three budget checkpoints \(N_F\le1002,3002,6002\).

## Key measured values

| alpha | best \(I_\perp\), \(N_F\le6002\) | lowest dense barrier, \(N_F\le6002\) (kcal/mol) |
|---:|---:|---:|
| 0.03 | 1.16216 | 28.2004 |
| 1 | 0.0460167 | 20.3567 |
| 10 | 0.0497513 | 20.4687 |
| 20 | 0.0628500 | 20.4688 |
| 100 | 0.126783 | 20.6992 |

The robust guide has its minimum near \(\alpha=5.02\) for \(I_\perp\) and
\(\alpha=16.7\) for the barrier. At \(\alpha=100\), the guide rises only to
0.1357 and 21.023 kcal/mol, respectively. Thus the plot shows a broad
medium-alpha well whose right side is visible but mild. The selected
\(\alpha=1\) observation itself is better than the fitted broad trend in both
panels.

## Large-alpha result

The new direct probes give:

| alpha | budget | minimum \(I_\perp\) | total forces at minimum | final \(I_\perp\) |
|---:|---:|---:|---:|---:|
| \(10^4\) | 300 calls/image | 0.0948829 | 2182 | 2.15473 |
| \(10^5\) | 300 calls/image | 0.0508498 | 5602 | 0.0837562 |
| \(10^4\) | 1000 calls/image | 0.0591820 | 10502 | 1.20738 |
| \(10^5\) | 1000 calls/image | 0.0508498 | 5602 | 0.0837562 |

At \(N_F\le6002\), their independently densified barriers are 20.6549 and
20.7336 kcal/mol. These results do not support portraying large alpha as
catastrophically divergent. They show that very large alpha can still reach
the low-residual, correct-barrier region, but its raw \(I_\perp\) history is
nonmonotone and can spend many evaluations away from its best attained state.
Above roughly \(10^2\), the proximal term is weak relative to the retained
trust cap, so the practical behavior approaches a cap-dominated saturation
regime.

## What the theorem does and does not establish

The article theorem applies to the exact, unreparameterized path-space
iteration at every fixed \(\alpha>0\), conditional on:

1. an exact global minimizer of every frozen-normal subproblem;
2. uniform \(H^2\) boundedness of the path sequence; and
3. a uniform nondegeneracy bound on path speed.

Under those assumptions it proves \(I_\perp\to0\). It does not prove an
alpha-independent convergence rate, monotonic decrease of \(I_\perp\), or
convergence of the finite-image implementation with a 15-evaluation inexact
inner solve and trust cap. The correct paper-level interpretation is:

> For every fixed positive alpha, the ideal exact IVM has conditional
> asymptotic convergence. In the practical finite-image solver, alpha controls
> efficiency until the weak-proximal, trust-cap-dominated saturation regime is
> reached.

This is consistent with the high-alpha probes and does not claim a practical
unconditional guarantee that the numerical experiment has not established.

## Audit

- 17 focus-range runs share initial path SHA-256
  `57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996`.
- Focus-range optimization evaluations: 98,749.
- Focus-range diagnostic evaluations: 21,403, excluded from \(N_F\).
- Additional \(10^4,10^5\) 300-call probe evaluations: 12,004.
- Additional high-alpha diagnostic evaluations: 2,518.
- Every run-level backend count equals the sum of image-evaluator cache misses.
- All tangents are normalized centered differences.
- All active-alpha arrays contain exactly their nominal fixed value.
- The two plotted versions share one raw CSV and one fitted-curve CSV.
- 53 core/cache/tangent/string-method tests pass.
- Both PDFs are one-page vector files with no raster images and all fonts
  embedded.

## Files

Figures and data:

`/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/figures/ivm_alpha_well_20260728a`

Run reports:

- `reports/ivm_fixed_alpha_well_fill_20260728a`
- `reports/ivm_fixed_alpha_large_probe_20260728a`
- `reports/ivm_fixed_alpha_large_long_20260728a`

Reproduction scripts:

- `src/run_ivm_fixed_alpha_proximal_only.py`
- `src/plot_ivm_alpha_well.py`
