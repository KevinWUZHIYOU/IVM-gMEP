# V15 frozen-frame IR-IVM development specification

Status: **frozen development design; not a formal registration or MD result**.

This document fixes the V15 controller before the first V15 grid trajectory is
created.  V15 changes two noise-decision branches of V14 and preserves the
projected equation, public-path semantics, cost horizon, parameter grid, and
post hoc scientific screen.

## Revision rationale, and nothing more

The completed V14 development analysis (SHA256
`fcbb44d0a90f0b4975de7196ff9664e6a5469d3f0fa447b25e82d7817e815c98`)
reported `NO_COMBINATION_PASSED` after all 48 exact trajectories.  Under the
calibrated synthetic-noise oracle, a symmetric residual-interval decision and
an unresolved H0-versus-L-BFGS comparison could repeatedly request exact-center
resampling even when a statistically admissible forward H0 step existed.

That observation is used only to state the controller defect addressed here.
It does not enter scoring or selection, and no V14 path, metric, diagnostic
rank, or visually preferred parameter changes the V15 factorial, seeds,
2,000-query-equivalent horizon, acceptance rules, passing order, or archived
current-IVM baseline.

## Frozen Riemannian normal equation

At a paid, article-reparameterized outer anchor \(A_k\), freeze the sampled
metric \(M_k\), \(G_k=M_k^{-1}\), the article centered tangent \(t_k\), and a
per-image basis \(B_k\) satisfying

\[
B_{k,i}^{\mathsf T}G_{k,i}B_{k,i}=1,
\qquad
B_{k,i}^{\mathsf T}G_{k,i}t_{k,i}=0.
\]

The basis gauge is fixed continuously by image and outer state.  For any paid,
article-reparameterized path \(X\) in the same trust neighbourhood, define

\[
z_i(X)=B_{k,i}^{\mathsf T}G_{k,i}
       \operatorname{per\_diff}(X_i,A_{k,i}),
\qquad
q_i(X)=B_{k,i}^{\mathsf T}\nabla F_i(X),
\]

and solve the frozen projected proximal root

\[
P_k(X)=q(X)+\frac{z(X)}{\alpha}=0.
\]

This remains a projected root on the article-reparameterized path manifold; it
is not called the exact gradient of a composed variational objective.

An internal proposal is lifted from the current path, passed through the
unchanged article reparameterizer, and only that selected path receives a
complete paid 20-image evaluation.  All later calculations use

\[
\delta=z_k(X_{j+1})-z_k(X_j),
\]

recomputed from the two actual post-reparameterization paths.  The nominal
pre-reparameterization proposal is never substituted for \(\delta\).

## Noise estimates and pooled exact-center observations

The ten mean-force block estimates are projected on the frozen basis.  For
equal blocks,

\[
\operatorname{SE}_i=
\sqrt{\frac{\sum_{b=1}^{10}(p_{ib}-\bar p_i)^2}{10(10-1)}}.
\]

For unequal retained-block sizes \(n_{ib}\), reconstruct the sizes exactly as
`np.array_split` applied to
`frame_counts[i] - discarded_frame_counts[i]`, and use

\[
\bar p_i=\frac{\sum_b n_{ib}p_{ib}}{N_i},\qquad
\operatorname{SE}_i^2=
\frac{\sum_b n_{ib}(p_{ib}-\bar p_i)^2}{(B-1)N_i},
\quad N_i=\sum_b n_{ib}.
\]

The same weights define aggregate gradient and metric estimates, gradient
covariances, and exact-center pooled estimates.  State 0 is one aggregate block
without a retained block ledger, so its SE is unavailable rather than zero.
A fixed safety multiplier \(\kappa=2.262\) defines

\[
L(P)=\lVert(|P|-\kappa\operatorname{SE})_+\rVert_2,
\qquad
U(P)=\lVert|P|+\kappa\operatorname{SE}\rVert_2.
\]

The multiplier is an algorithmic safety constant, not a claim of sequential
confidence coverage.

If an outer anchor has \(L(P_0)=0\), the controller holds the exact same path,
purchases another complete band, and pools the independent estimates.  During
this `ANCHORING` phase, every pooled observation rebuilds the provisional
tangent, \(G\), and basis.  The anchor becomes `READY` atomically only after
its lower bound is positive, so no stale sampled geometry is inherited.

A resample reuses the exact path bytes, path hash, and original article
reparameterization certificate.  It does not call the reparameterizer again.
It costs 20 query equivalents, becomes a public state, does not increment the
internal movement count, and cannot create a zero-displacement secant pair.

## V15 one-sided confident relative-residual stop

After at least one movement in a frozen outer problem, V15 stops that inner
solve if and only if

\[
U(P_j)\le rL(P_0).
\]

This is the sole confidence-protected \(r\)-stop.  If the inequality is false,
including when the current and target uncertainty intervals overlap, the stop
has not been established.  Provided the internal movement count is below
\(m_{\max}\), the controller continues with an admissible movement in the same
frozen outer problem.  It does not buy an exact-center resample merely to
resolve \(r\).  The field `residual_interval_ambiguous` is retained only as a
diagnostic, and a continued movement records
`inner_stop_reason="continue_same_frozen_outer_r_stop_not_resolved"`.

At \(m_{\max}\), a satisfied one-sided stop still takes precedence.  Otherwise
the controller refreshes the outer frame without buying a decision-only band.
Thus ambiguity can persist only until a clear stop or the fixed integer cap;
it cannot become an absorbing HOLD loop.

## V15 pre-takeover direction decision

Before the global warm takeover, H0 is the safe default direction.  Whenever
enough valid reconstructed pairs make an L-BFGS proposal available, its
admissibility score is compared with H0 using their shared-source block
uncertainty.  Let \(D\) be the L-BFGS-minus-H0 score advantage and
\(\operatorname{SE}_D\) its standard error.  A warm takeover first requires
that the positive advantage be resolved by the already fixed secant margin,

\[
D>z_{\rm secant}\operatorname{SE}_D,
\qquad z_{\rm secant}=2.
\]

The public boolean `lbfgs_score_advantage_resolved_positive` records this
resolved-\(D\) test.  It is necessary but not sufficient for a committed
takeover: the selected L-BFGS movement must also have a strictly positive own
improvement lower confidence bound.  If the resolved-advantage candidate fails
that own-LCB gate with positive reducible uncertainty, V15 HOLDs,
`strict_lbfgs_win_trigger` remains false, and the global latch remains unset.
Thus `strict_lbfgs_win_trigger` records the accepted, latching takeover rather
than merely the resolved-advantage prerequisite.

If that inequality is not established, including an ambiguous difference, H0
is the required fallback.  V15 moves with H0 only when H0 is valid and its own
improvement lower confidence bound is strictly positive.  If the selected H0
lower bound is nonpositive and its uncertainty is positive and reducible, the
controller may HOLD.  The nonselected L-BFGS candidate's own lower bound is not
consulted to bypass an unresolved takeover comparison: without a resolved
positive advantage, L-BFGS cannot take over.  With zero SE, a score tie goes to
H0.  If the required fallback is invalid and uncertainty cannot justify the
specified HOLD branch, the state becomes durable `TERMINAL` without an
additional oracle call.

The run manifest fixes
`relative_residual_ambiguous_action="continue_movement_same_frozen_outer"`.
These policies apply only before takeover; they do not reintroduce H0 after a
warm takeover.

## Curvature information and one-way warm takeover

Keep at most 16 distinct, complete, post-reparameterization public path
observations.  When an outer frame changes, reproject the applicable local
suffix into the new \((A_k,B_k,G_k)\) frame; do not reuse old reduced secants as
if the operator were unchanged.  For two observations in the same rebuilt
frame,

\[
s=z_b-z_a,
\qquad
y=P_b-P_a=B_k^{\mathsf T}(g_b-g_a)+\frac{s}{\alpha}.
\]

The \(s/\alpha\) term occurs exactly once.  A pair is retained only if paths
are distinct, both endpoints lie within the existing per-image trust radii of
the new anchor, the reprojected \(\lVert s\rVert\) exceeds
`secant_norm_floor`, and the pair passes the existing cosine and block-noise
curvature gates.  At least two qualifying pairs are required for L-BFGS.

Takeover is global and one-way.  Once a resolved positive L-BFGS advantage
latches it, H0/cold selection is forbidden in all later outers.  A rebuilt
frame uses L-BFGS only with at least two reconstructed pairs.  Otherwise it
uses the latest valid scalar

\[
\gamma=\frac{s^{\mathsf T}y}{y^{\mathsf T}y},
\]

clipped to the unchanged interval
\([0.25\alpha,4\alpha]\), with its source state persisted.  This is labelled a
warm scalar safeguard, not current-frame L-BFGS memory.  An inadmissible warm
safeguard causes HOLD only when uncertainty is reducible; otherwise the run
terminates without cold recovery.

## Public state, query cost, and restart requirements

The explicit phases remain `ANCHORING`, `READY`, `MOVE_PREPARED`,
`RESAMPLE_PREPARED`, `HOLD`, and `TERMINAL`/`FAILED`.  Every complete band is
published immediately.  A complete band has exactly 20 internal images and
costs 20 development query equivalents.  Both components returned at one 2D
collective-variable point constitute one evaluation.  No unselected path may
be evaluated.  Interrupted image evaluations remain in the absolute ledger
but do not form a public band; resume neither hides nor double-counts them.

Prepared and `FAILED` phases are durable in status.  A zero-query `TERMINAL` is
durable in status and completion; resume validates the public trajectory and
returns the same terminal result without regenerating a proposal or calling
the oracle.  An outer-boundary refresh is a zero-query atomic checkpoint
because it changes \(A,B,G,z,P\), eligible secants, and the next direction.
Deterministic replay validates every durable public state and the source hash.

The historical shared initial observation remains the common causal input at
incremental cost zero.  It has no original ten block means and uses exactly one
`initial_no_se_special_case`: SE is unavailable, no noise-resolved stopping or
curvature claim is made, and exactly one frozen initial-scale proposal is
article-reparameterized and evaluated as the first 20-query public state.  All
later confidence decisions require genuine block estimates.

Development query equivalents are not formal mean-force evaluations.  This
screen contains no restrained-MD execution path and cannot authorize one.

## Unchanged systematic development mesh and screen

The V15 screen repeats the complete V14 factorial without alteration:

\[
\alpha\in\{10,13,16\}\times10^{-5},\qquad
r\in\{0.4,0.7\},\qquad
m_{\max}\in\{4,8\}.
\]

The 12 combinations are each run once on the deterministic development oracle
and once on each fixed synthetic-noise stream 1401, 1402, and 1403, yielding
48 exact trajectories.  Every run uses the same common initial path, costs
0--2,000 query equivalents in exact increments of 20, and is scored blindly by
the same deterministic posterior scorer.

The pointwise 20--1,600 rules; 1,620--2,000 median, normalized-AUC,
win-fraction, and individual-seed rules; deterministic guard; correct-path
terminal gates; archived current-IVM selection baseline; and passing order are
unchanged.  Archived FENEB remains contextual only.  If no combination passes,
the analyzer makes no recommendation.

## Required tests before any V15 trajectory

1. Frozen basis, periodic coordinate, gauge, and actual
   post-reparameterization displacement identities.
2. The \(P\) and \(y\) equations, with no duplicated \(s/\alpha\).
3. Ten-block standard error and weighted pooling against analytic examples.
4. Exact-center resampling preserves path bytes/hash/certificate, costs 20,
   does not increment movement, and cannot create an \(s=0\) pair.
5. A satisfied one-sided \(U(P_j)\le rL(P_0)\) stops; an overlapping or
   otherwise unresolved interval continues movement below \(m_{\max}\).
6. An unresolved L-BFGS advantage selects positive-LCB H0.  A resolved positive
   advantage latches takeover only after the selected L-BFGS own-LCB gate;
   failure of that gate with reducible uncertainty HOLDs without latching.
7. One-way warm takeover forbids H0 thereafter and accurately labels the
   bounded scalar safeguard.
8. Engineered \(r\)-stop and \(m_{\max}\)-refresh cases create different next
   paths and persisted reasons.
9. One complete band equals 20 query equivalents; partial-band ledger, exact
   resume, source validation, and budget-prefix invariance are explicit.
10. No hidden oracle calls, no evaluation of unselected candidates, and no
    access to article performance metrics during optimization.
