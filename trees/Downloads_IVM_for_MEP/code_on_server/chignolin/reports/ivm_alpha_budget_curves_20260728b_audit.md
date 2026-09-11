# Chignolin fixed-alpha connected budget curves audit

## Correction

The preceding robust quadratic guide is not used in this revision.

For each of the three total-force budgets,
\[
N_F\le1002,\qquad N_F\le3002,\qquad N_F\le6002,
\]
the independently measured values are sorted by alpha and connected directly
with ordinary line segments. There is:

- no regression;
- no smoothing;
- no interpolation between unmeasured alpha values other than the displayed
  straight segment connecting adjacent observations;
- no omission of the poorer \(\alpha=2,3,5\) observations between
  \(\alpha=1\) and 10.

The same three raw curves are used in the log-y and linear-y figures. Only the
vertical-axis scale changes.

## Metrics

For each alpha and displayed budget:

- Panel a is the minimum unnormalized \(I_\perp\) actually attained at or below
  that total-force budget.
- Panel b is the lowest independently densified raw 0 K forward barrier
  measured at or below that total-force budget.

The cost \(N_F\) is the sum of joint energy-force cache misses over all 22
images. Independent densification evaluations are excluded from \(N_F\).

The displayed alpha interval is \(0.03\) to 100 and contains 17 independent
fixed-alpha runs. The separate \(10^4\) and \(10^5\) saturation probes remain
in the accompanying CSV/report but are not connected across the large
unmeasured gap.

## Provenance and audit

- Common initial-path SHA-256:
  `57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996`.
- Every run uses fixed alpha, centered-difference tangents,
  `state_globalization=proximal_only`, and no outer \(I_\perp\) guard.
- Focus-range optimization evaluations: 98,749.
- Focus-range diagnostic evaluations: 21,403, excluded from optimization cost.
- Every backend evaluation count equals the sum of the image-evaluator cache
  misses.
- All three plotted budget series come directly from the archived CSV.
- 53 core/cache/tangent/string-method tests pass.
- Both PDFs are one-page vector files with embedded fonts and no raster images.

## Output

Figures and data:

`/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/figures/ivm_alpha_budget_curves_20260728b`

Reproduction script:

`/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/src/plot_ivm_alpha_well.py`
