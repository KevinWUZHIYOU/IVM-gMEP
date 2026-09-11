# Ala2D IR-IVM V14 development-screen diagnosis

## Status and scope

This is a development-only diagnosis. It is not formal restrained-MD evidence,
does not recommend a parameter combination, and does not authorize replacing
either paper figure. The frozen V14 screen completed all 48 registered
trajectories at the exact incremental query-equivalent checkpoints
`0, 20, ..., 2000`:

- 12 parameter combinations:
  `alpha = {10e-5, 13e-5, 16e-5}`,
  `r = {0.4, 0.7}`, and `m_max = {4, 8}`;
- one deterministic free-energy-surface trajectory and three fixed calibrated
  noise repetitions (`1401`, `1402`, and `1403`) per combination;
- 96,000 development query equivalents in total. These are not new
  restrained-MD mean-force evaluations and are not formal NoF.

The preregistered analysis decision is `NO_COMBINATION_PASSED`. Four
combinations passed the terminal correct-path guard in all four repetitions:

- `alpha=10e-5, r=0.7, m_max=4`;
- `alpha=10e-5, r=0.7, m_max=8`;
- `alpha=13e-5, r=0.4, m_max=4`;
- `alpha=16e-5, r=0.4, m_max=4`.

None passed the full front/middle, noisy-tail, and deterministic guards. The
analysis program labels `alpha=13e-5, r=0.4, m_max=4` as a diagnostic example
only; it is explicitly not a recommendation.

## What the curves show

With the deterministic free-energy-surface oracle, all 12 trajectories made
100 movements, made no exact-center HOLD, passed the deterministic tail gate,
and reached a terminal median \(I_\perp^G\) of approximately
\(3.4\times10^{-11}\). Thus the frozen-frame Riemannian update and article
reparameterization can converge accurately when the mean force is exact.

Under calibrated Ala2D mean-force noise, the 36 trajectories contain 3,600
paid bands but only 328 movements. The remaining 3,272 bands (90.89%) repeat
the identical collective-variable path:

- 2,858 HOLDs (87.35% of all HOLDs) arise from ambiguity in the relative
  residual interval used by `r`;
- 414 HOLDs (12.65%) arise only because the H0 and L-BFGS admissibility scores
  cannot be distinguished statistically.

The closest terminal correct-path failure is `alpha=10e-5, r=0.7`; its
`m_max=4` and `m_max=8` trajectories are byte-for-byte identical because the
`m_max` limit is never reached. On the common deterministic post-hoc scorer,
its three-repetition noisy-tail median is:

| metric | V14 | archived Current IVM |
|---|---:|---:|
| \(I_\perp^G\) | 4.457743 | 2.157923 |
| \(|\Delta F^\ddagger-\Delta F^\ddagger_*|\) (kcal mol\(^{-1}\)) | 0.040877 | 0.037830 |

The archived FENEB curve is contextual rather than the selection baseline. Its
recorded tail median \(I_\perp^G\) is 1.253204. It therefore reinforces the
visual observation that the unresolved problem is the noisy late-stage normal
accuracy, not path spacing or deterministic convergence.

## Structural cause

V14 declares a confident relative-residual stop only when

\[
U_{\mathrm{current}} \le r L_{\mathrm{anchor}},
\]

and a confident continuation only when

\[
L_{\mathrm{current}} > r U_{\mathrm{anchor}}.
\]

When neither inequality holds, V14 repeatedly samples the current path. This
shrinks the current interval but leaves the outer-anchor interval fixed. A
current residual can therefore remain forever inside the nonshrinking interval
\([rL_{\mathrm{anchor}},rU_{\mathrm{anchor}}]\). In 34 of the 36 noisy
trajectories the terminal state is this ambiguity HOLD. The median final HOLD
segment is 90 paid bands even though the current-path standard-error norm has
fallen to 0.167 of its value at the beginning of that segment.

A representative trajectory (`alpha=13e-5, r=0.7, m_max=4`, repetition 1401)
moves only at incremental query equivalents 20 and 40. It then spends the next
98 bands at exactly the same path. Its fixed scaled anchor interval is
`[94.3, 127.1]`; the current interval shrinks from `[99.9, 120.8]` at 60 to
`[106.2, 109.7]` at 2000 but can never leave that dead zone.

Candidate arbitration is a second, removable cost. For every one of the 414
H0/L-BFGS ambiguity HOLDs, H0 is geometrically valid and its own admissibility
lower confidence bound is positive (minimum 0.01103). Those observations do
not justify waiting; they only fail to prove that L-BFGS is better than H0.

The noise oracle itself is not the immediate cause. Fresh synthetic projected
standard errors are only 1.022 times the matched archived source values at the
median. Spatial support is inside the registered limits. The center residual
does include systematic restrained-MD versus surrogate mismatch; it is about
2.1--2.2 times larger than a pure random-innovation interpretation, but it is
appropriate for the registered total-discrepancy stress test. Lowering its
noise scale would hide the controller deadlock rather than repair it.

## Predeclared V15 revision rationale

The next development algorithm should change decisions, not cherry-pick a new
parameter:

1. Treat `r` as a one-sided confident stop. Rebuild the outer frame only after
   the stop inequality is proved. Otherwise continue the frozen-inner movement
   until a later confident stop or `m_max`; never spend a band solely to decide
   the `r` ambiguity.
2. Before one-way takeover, select L-BFGS only when its positive advantage over
   H0 is statistically resolved. Otherwise use valid H0 when H0's own
   admissibility lower bound is positive; never spend a band solely to
   arbitrate the two methods.
3. Preserve exact-center resampling for genuine low-signal or nonpositive
   movement evidence, together with the existing trust, reparameterization,
   NoF, checkpoint, and replay invariants.

V15 must be frozen before its first result and rerun the same complete 12-way
factorial screen, deterministic trajectory, three fixed noise repetitions,
2000-query-equivalent horizon, and acceptance rules. V14's failed trajectories
cannot be reclassified as a V15 pass.

## Machine-verifiable provenance

- frozen plan SHA256:
  `2399bfa55445374644d72faec09c29a6b645b31b9fd081faddbc90022a527dd8`;
- V14 core SHA256:
  `01cdf6435f25b6e2e9b7527fb9bcc960565662a2889d8f42cf83cd69375381a1`;
- execution manifest SHA256:
  `be4bb8274aef3e5ea4a728025088185e88b4177992a5d4dc8bfadcbad7941d9d`;
- complete analysis JSON:
  `reports/development_screen_analysis.json`, SHA256
  `fcbb44d0a90f0b4975de7196ff9664e6a5469d3f0fa447b25e82d7817e815c98`;
- summary CSV SHA256:
  `c664e88b61ac4432fcc4adbdb01dfc2ceaeaa4de0615e9b4e6d227e8df2166e6`;
- long-form metric CSV SHA256:
  `dc2d5c875a3aedbfffb9be7fb634b42f0143c61958b8c824d43f581e627b192d`;
- parameter-comparison PNG SHA256:
  `e2a9febad9f5998d52e89a86c8d369e3ec1485088ff1395dd52a649c6ccd0156`;
- diagnostic full-curves PNG SHA256:
  `79f05fe79d7c87a1f3d5d606c54c4442c0861a876b225cc85a1f49b03a6dd9fb`.

The formal paper PDFs were not modified. Their SHA256 values remain
`ed57a7089730e760862ac435545d7e9fc8534601ee8c9008635b1a16d54b9932`
and
`e872be9fd9d4cea83df67f1f94fe04e6a06ff3d5093108a4afc07877f8b993e1`.
