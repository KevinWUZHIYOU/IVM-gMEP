# V15 Ala2D development-screen failure diagnosis

Status: **development screen incomplete; no parameter selection and no formal-MD promotion**.

This report registers the terminal state of the frozen V15 development screen.
It is a failure diagnosis, not a replacement analysis and not formal evidence.
The machine-readable companion is
`reports/V15_DEVELOPMENT_SCREEN_FAILURE_DIAGNOSIS.json`.

## Integrity and decision

The existing validation path independently checked all 48 run bindings,
completion records, public-state ledgers, and the SHA256 of every ledger-bound
state file.  All checks passed:

- 48 registered trajectories and 48 completion records;
- 3,210 verified public state files;
- 30 trajectories contain the complete exact sequence
  (0,20,\ldots,2000);
- 18 trajectories ended through the registered algorithmic zero-query stop;
- zero execution failures, corrupt ledgers, missing states, or other statuses;
- 63,240 development query equivalents were used in total.

The verified bindings are:

- frozen plan SHA256:
  `38c9c8aee4974a692949ce5b665979cfba12453562c7d4e303d18cf49f4cbd4c`;
- execution-manifest SHA256:
  `bdd531d70d540bbf5fa0e9d5e3394fbf5f52d149b85ff85ecb17126fb9dbaac7`;
- complete validated-inventory digest:
  `6efa8aa9560cb69df73279d5d1ce75ae935fcb94e0ec710e0e24244a2805d0d7`;
- V15 core SHA256:
  `07ff2d70216bae7d3f95997805835b652bc755ab8f16cb4e82a489ae69ba56dc`;
- registered analyzer SHA256:
  `9c4fc954cc141ab67f844fa479151882a8d6c2d0cdf48d8c920722c3453ce3f6`.
- analyzer-generated nonselecting inventory report SHA256:
  `08ad79068d9a2c0b041c4a645c5867d33ac22fab5514153c361bbf4014c8939b`.

The nonselecting report is
`reports/development_screen_analysis.json`; it records status
`INCOMPLETE_DEVELOPMENT_SCREEN`, `decision=null`, and no recommended or
diagnostic parameter.

The registered analyzer returned exit code 2 with the exact refusal:

> ERROR: development screen is incomplete: 30/48 runs contain every exact
> checkpoint through 2000; LOCF and selection are forbidden

Consequently no missing suffix may be filled by carrying forward the last
path, no parameter combination may be ranked or recommended, and this screen
cannot authorize a formal restrained-MD run.

The 63,240 count is **development query equivalents**, not formal Number of
mean-force evaluations.  A complete development band evaluates all 20
internal collective-variable points and costs 20 equivalents; the two
mean-force components returned at one point jointly count as one evaluation.

## Completion by force stream

| force stream | complete through 2000 | algorithmic stop | trust-region exit | cap implementation failure | development equivalents |
|---|---:|---:|---:|---:|---:|
| deterministic | 12 | 0 | 0 | 0 | 24,000 |
| synthetic seed 1401 | 5 | 7 | 4 | 3 | 11,060 |
| synthetic seed 1402 | 7 | 5 | 5 | 0 | 14,820 |
| synthetic seed 1403 | 6 | 6 | 3 | 3 | 13,360 |
| **total** | **30** | **18** | **12** | **6** | **63,240** |

Every deterministic trajectory completed.  All 18 early stops occurred in a
noisy trajectory before the global L-BFGS takeover: `globally_latched=false`,
the failure was recorded on H0, and `warm_error=none` in every case.  Therefore
the incomplete factorial is caused first by candidate globalization under
noise, independently of the later warm-HOLD defect.

The two terminal failure modes are:

1. Twelve paths failed because article reparameterization placed at least one
   image outside the frozen-anchor trust radius.  V15 caps the proposed step
   norm, but it does not include the current displacement already accumulated
   from the frozen anchor when calculating the remaining radius.
2. Six paths failed because a single binary64 `nextafter` correction did not
   make the whole-band common cap satisfy every image radius.  This is a
   numerical implementation/globalization failure, not evidence that the
   trajectory entered a wrong physical channel.

## Completion by parameter combination

Each row includes the deterministic run and the three noisy streams.

| (alpha) | (r) | (m_{\max}) | complete | stopped | stop details |
|---:|---:|---:|---:|---:|---|
| (1.0\times10^{-4}) | 0.4 | 4 | 3 | 1 | cap: seed 1403 at 160 |
| (1.0\times10^{-4}) | 0.4 | 8 | 4 | 0 | -- |
| (1.0\times10^{-4}) | 0.7 | 4 | 2 | 2 | trust: 1402 at 220; cap: 1403 at 160 |
| (1.0\times10^{-4}) | 0.7 | 8 | 2 | 2 | trust: 1401 at 300, 1402 at 220 |
| (1.3\times10^{-4}) | 0.4 | 4 | 1 | 3 | trust: 1401/1402 at 100; cap: 1403 at 220 |
| (1.3\times10^{-4}) | 0.4 | 8 | 2 | 2 | cap: 1401 at 100; trust: 1402 at 180 |
| (1.3\times10^{-4}) | 0.7 | 4 | 2 | 2 | trust: 1401/1402 at 100 |
| (1.3\times10^{-4}) | 0.7 | 8 | 3 | 1 | cap: 1401 at 100 |
| (1.6\times10^{-4}) | 0.4 | 4 | 3 | 1 | trust: 1403 at 300 |
| (1.6\times10^{-4}) | 0.4 | 8 | 3 | 1 | cap: 1401 at 140 |
| (1.6\times10^{-4}) | 0.7 | 4 | 3 | 1 | trust: 1403 at 260 |
| (1.6\times10^{-4}) | 0.7 | 8 | 2 | 2 | trust: 1401 at 220, 1403 at 260 |

No parameter row is complete merely because its deterministic trajectory is
well behaved.  The noisy failure pattern is non-monotone in (alpha), (r),
and (m_{\max}), so the incomplete table cannot support a causal tuning claim.

## Seed-1401 trajectory diagnosis

Seed 1401 was inspected before the complete factorial ended and is therefore
design evidence only.  Its 11,060 development equivalents comprise 333
movement bands and 220 exact-center HOLD bands, or 6,660 versus 4,400
equivalents.  The HOLD causes were:

- 110 candidate-improvement lower bounds that were not positive;
- 108 invalid warm candidates with nominally reducible uncertainty;
- 2 unresolved outer-anchor lower bounds.

Two post-takeover paths entered absorbing behavior:

- ((10^{-4},0.7,4)): 38 consecutive HOLDs from 1260 through 2000; its last
  movement was at 1240.
- ((1.6\times10^{-4},0.7,4)): 70 consecutive HOLDs from 620 through 2000;
  its last movement was at 600.

The standard error decreased during both sequences, but the warm candidate
continued to have nonpositive post-reparameterization improvement.  Merely
testing whether any standard error remains positive therefore does not show
that another exact-center sample can repair the direction.

The following numbers use the same deterministic post-hoc scorer at exact
shared checkpoints, with no last-observation carry-forward.  They are a
diagnostic of the five complete seed-1401 paths, not the registered
three-repetition selection.

| (alpha,r,m_{\max}) | correct MFEP guard | front/middle joint wins / 80 | tail (I_\perp) wins | tail (I_\perp) AUC ratio | tail barrier-error wins | tail barrier-error AUC ratio |
|---|---|---:|---:|---:|---:|---:|
| (10^{-4},0.4,4) | pass | 3 | 14/20 | 0.914 | 6/20 | 1.007 |
| (10^{-4},0.4,8) | pass | 0 | 0/20 | 3.372 | 0/20 | 1.175 |
| (10^{-4},0.7,4) | pass | 1 | 1/20 | 1.327 | 7/20 | 0.99997 |
| (1.6\times10^{-4},0.4,4) | pass | 3 | 3/20 | 1.196 | 7/20 | 0.99967 |
| (1.6\times10^{-4},0.7,4) | pass | 2 | 0/20 | 1.850 | 3/20 | 1.016 |

All five complete paths stayed in the registered MFEP channel and won all 20
tail (I_{\rm arc}) comparisons.  Thus path regularity is not the limiting
observable.  No seed-1401 trajectory simultaneously passed the tail
(I_\perp) and barrier-error requirements, and a single inspected stream
cannot determine the required three-stream median.

## Minimum V16 correction directions

V15 does not justify a parameter recommendation.  Its failures instead fix
two minimal, orthogonal algorithm changes for a separately frozen V16.

### 1. Anchor-aware trust globalization

For current path (X), frozen outer anchor (A), basis (B_i), reduced
direction (p_i), and radius (ho_i), set

\[
d_i=\operatorname{per\_diff}(X_i,A_i),\qquad v_i=B_i p_i.
\]

Choose one common scalar (s\in[0,1]) from the positive quadratic roots of

\[
\lVert d_i+s v_i\rVert^2\le \rho_i^2
\]

for every image.  This accounts for the already-used part of the trust
radius.  Apply an explicit binary64 inward safety margin.  After article
reparameterization, allow only a fixed, preregistered finite sequence of
whole-path common-scale backtracks, for example (s,s/2,s/4); no image-wise
cap is allowed.  Accept only if the reparameterization certificate is valid
and every actual post-reparameterization image remains inside its radius.

If the finite sequence fails, the same exact current-path hash may trigger at
most one zero-query outer-frame refresh.  A second failure at the same path is
durably terminal.  Candidate construction, reparameterization, common-scale
backtracking, and the outer refresh cost zero; only the selected complete
20-image evaluation costs 20 equivalents.

### 2. One independent warm validation and bounded warm restart

After global takeover, when the candidate-improvement lower bound is not
positive, V16 must freeze the candidate path, its actual
post-reparameterization normal displacement, the outer frame, and the
candidate hash.  It may purchase exactly one new exact-center complete band
as independent validation.  The validation score must use the fresh raw
projected-residual blocks for that already frozen displacement, rather than
regenerating a new adaptive candidate from the same data.

With the already frozen multiplier (kappa=2.262), distinguish:

- lower bound (>0): validate and evaluate the frozen movement;
- upper bound (le0): confirmed non-improving warm direction;
- interval still crossing zero: unresolved after the single registered
  validation, not a confirmed bad direction.

The last two outcomes both permit a bounded warm restart, but their labels
must remain distinct.  The global one-way takeover latch remains true and H0
motion remains forbidden.  Clear current L-BFGS pairs and set a persistent
curvature-epoch floor so no observation before the reset can reconstruct an
old pair on a later outer refresh.  Refresh the outer frame from the current
already-paid, pooled metric and mean force at zero cost.  Until two new
noise-qualified secants exist, only a persisted warm scalar safeguard may
move the path.

At most one warm restart is permitted per exact current-path hash; a repeated
failure without an intervening movement is durably terminal.  The independent
validation band costs exactly 20; its HOLD does not increment the internal
movement count and cannot create a zero-displacement secant.  The subsequent
selected movement, if any, costs another 20.

The V15 streams, especially seed 1401, have now influenced the V16 design.
They cannot be reused as independent V16 confirmation evidence.  V16 must be
frozen first and evaluated on newly preregistered, previously unseen noisy
streams before any formal-MD decision.

## Formal artifacts

No source program and no formal PDF was modified by this diagnosis.  The two
requested formal figures retain their registered SHA256 values:

- `MFEP_IVM_vs_FENEB_bestpair_threepanel.pdf`:
  `ed57a7089730e760862ac435545d7e9fc8534601ee8c9008635b1a16d54b9932`;
- `MFEP_IVM_vs_FENEB_contour_fourpanel.pdf`:
  `e872be9fd9d4cea83df67f1f94fe04e6a06ff3d5093108a4afc07877f8b993e1`.
