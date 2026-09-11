# Unified V2 fixed-alpha scoring preregistration

This scoring rule is frozen before any unified V2 production result or dense
V2 barrier file exists.

## Budgets and cost axis

The four budgets are total Number of Forces
\(B\in\{1000,3000,6000,10000\}\), computed as
`counter_vectors.sum(axis=1)` over all 22 images. Fixed-endpoint first
evaluations are included. Diagnostic evaluations are excluded.

For each run and each requested budget, the actual endpoint is the final
measured optimization state satisfying \(N_F\le B\). The trailing window is
the last 30% of that actually attained endpoint budget. Endpoint values are
always reported next to trailing statistics.

## Primary stationarity statistic

For every budget, the primary stationarity statistic is the arithmetic mean
of the unnormalized \(I_\perp\) over all measured optimization records in the
trailing window. No running minimum, best-so-far envelope, regression, or
interpolation is used in the fixed-alpha sensitivity figure.

The across-budget stationarity score is the geometric mean of the four
trailing arithmetic means. Smaller is better.

## Dense energy-barrier statistic

The sparse 22-image barrier is ineligible for scoring.

At seven evenly spaced target costs in every trailing window, adjacent path
segments are linearly densified and evaluated with a separate real OpenMM
CPU/double evaluator:

1. 10 subdivisions per segment;
2. 20 subdivisions per segment;
3. double to 40 and at most 80 when successive barriers differ by more than
   0.05 kcal/mol.

The diagnostic cache-miss count is audited separately and excluded from
\(N_F\).

The barrier reference is frozen at
\[
\Delta E_{\rm ref}=20.373739515742795\ {\rm kcal\,mol^{-1}},
\]
the adaptively densified final barrier of the preserved pre-V2 cap=15,
fixed-\(\alpha=1\) production run. It is not recomputed from, or changed
after inspecting, the new alpha sweep.

For each budget, the energy score is the RMS dense-barrier error over the
seven trailing paths:
\[
S_E(B)=\sqrt{\frac{1}{7}\sum_{j=1}^{7}
  \left(\Delta E_{\rm dense}^{(j)}-\Delta E_{\rm ref}\right)^2}.
\]
The raw seven barriers, their arithmetic mean, temporal standard deviation,
10-to-20 difference, and adaptive subdivision level are also reported.

The across-budget energy score is the geometric mean of the four
\(S_E(B)\) values. Smaller is better.

## Parameterization statistic

The endpoint equal-speed integral
\[
I_{\rm eq}=\int_0^1
(\varphi'(s)\cdot\varphi''(s))^2\,ds
\]
and segment-length coefficient of variation are reported without
best-so-far selection. Central-Hermite relocking must use no OpenMM query.

## Production selection

A candidate is production-eligible only if all of the following hold:

1. common initial-path SHA-256 and central-difference tangent are exact;
2. alpha is fixed for the full run and no alpha-specific option differs;
3. backend calls equal the sum of the 22 image cache misses;
4. every requested API/counter identity holds and blocked misses are zero;
5. the final \(I_\perp\), dense-barrier RMS error, and \(I_{\rm eq}\) are not
   worse than the preserved cap=15 production reference by more than 5%;
6. the candidate reaches the strict common precision attained by ODE-NEB
   and OBSM-FIRE.

Among eligible candidates, selection is lexicographic:

1. smallest across-budget stationarity score;
2. smallest across-budget dense-barrier score;
3. smallest final \(I_{\rm eq}\);
4. smallest total Number of Forces.

If no candidate satisfies the 5% non-inferiority gate, the unified V2
strategy is rejected rather than relabeling a weaker run as production.

## Single-well assessment

The sensitivity plot uses the measured alpha grid connected in increasing
alpha order, with markers at every actual alpha and no fitted regression.
Both endpoint and trailing-window variants are exported. A “single well”
claim requires one global basin around the selected moderate alpha in the
trailing statistics at all four budgets; local nonmonotonic points and any
secondary basin are shown and discussed rather than removed.
