# V14 frozen-frame IR-IVM development specification

Status: **development design, not a formal registration**.

This specification translates the first-round failure into testable numerical
semantics for the second-round Ala2D IR-IVM.  It does not authorize a paper
figure replacement or establish performance.

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

This is called a projected root on the article-reparameterized path manifold;
it is not described as the exact gradient of a composed variational objective.

An internal proposal is first lifted from the current path, then passed through
the unchanged article reparameterizer, and only that selected path receives a
complete paid 20-image evaluation.  All subsequent calculations use

\[
\delta=z_k(X_{j+1})-z_k(X_j),
\]

recomputed from the two actual post-reparameterization paths.  The nominal
pre-reparameterization proposal is never substituted for \(\delta\).

## Noise estimates, movement, and resampling

The ten mean-force block estimates are projected on the frozen basis.  For
equal blocks,

\[
\operatorname{SE}_i=
\sqrt{\frac{\sum_{b=1}^{10}(p_{ib}-\bar p_i)^2}{10(10-1)}}.
\]

For unequal retained-block sizes (n_{ib}), reconstruct the sizes exactly as
`np.array_split` applied to
`frame_counts[i] - discarded_frame_counts[i]`, and use

\[
\bar p_i=\frac{\sum_b n_{ib}p_{ib}}{N_i},\qquad
\operatorname{SE}_i^2=
\frac{\sum_b n_{ib}(p_{ib}-\bar p_i)^2}{(B-1)N_i},
\quad N_i=\sum_b n_{ib}.
\]

The same weights define aggregate gradient/metric estimates, gradient
covariances, and exact-center pooled estimates.  Equal sizes reduce exactly to
the formula above.  State 0 is strictly one aggregate block without a retained
block ledger and remains SE-unavailable.  A fixed safety multiplier
\(\kappa=2.262\) defines

\[
L(P)=\lVert(|P|-\kappa\operatorname{SE})_+\rVert_2,
\qquad
U(P)=\lVert|P|+\kappa\operatorname{SE}\rVert_2.
\]

The multiplier is an algorithmic safety constant, not a claimed sequential
confidence interval.

- If the outer anchor has \(L(P_0)=0\), hold the exact same path and purchase a
  new complete band, then pool the independent estimates.
- Otherwise retain both anchor bounds.  After at least one movement,
  \(U(P_j)\le rL(P_0)\) is a clear relative-residual stop, whereas
  \(L(P_j)>rU(P_0)\) is a clear continuation decision.  The interval between
  these two tests is a hold-and-resample decision.  This symmetric rule is the
  noise-protected interpretation; comparing both sides only with \(rL(P_0)\)
  would be an unlabelled point-estimate heuristic.
- A candidate step with predicted frozen-model improvement \(\Psi>0\), but
  noise-adjusted lower bound \(\Psi_{\rm LCB}\le0\), also triggers hold and
  resampling rather than movement.
- Before freezing an outer anchor, exact-center resamples continue until its
  lower bound is nonzero and any preregistered precision requirement is met.
  For an ambiguous state below \(m_{\max}\), resampling continues rather than
  forcing a move.  At \(m_{\max}\), an unambiguous residual stop takes
  precedence; otherwise the outer frame is refreshed without buying an extra
  decision-only band.

A resample must reuse the exact path bytes, path hash, and original article
reparameterization certificate.  It must not call the reparameterizer again.
It costs 20 NoF, becomes a public state, does not increment the internal
movement count, and cannot create a zero-displacement secant pair.

HOLD is allowed only for positive, reducible uncertainty.  A zero-SE
candidate-score tie follows the fixed deterministic pre-takeover rule (H0 wins
a tie).  If no admissible motion exists and uncertainty cannot be reduced, the
state becomes durable `TERMINAL`; the initial no-SE exception cannot bootstrap
into HOLD.

While an anchor is `ANCHORING`, each exact-center resample pools both gradient
and metric blocks, rebuilds the provisional tangent, (G), and basis from the
pooled estimates, and recomputes bounds in that frame.  It becomes `READY`
atomically only when the lower bound is positive, so an unresolved anchor
never retains stale sampled geometry.

## Curvature information and one-way takeover

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
are distinct, locality and normal-retention tests pass, and its curvature is
positive after a block-noise margin.  At least two qualifying pairs are needed
for the one-way L-BFGS takeover.  Before takeover the direction is the frozen
initial-scale direction.  After takeover, cold recovery is forbidden.  If a
new outer frame cannot reconstruct valid pairs, the method must label and use
only a bounded scalar safeguard; it must not claim that current-frame L-BFGS
memory exists.

The locality/normal-retention test adds no new ratio or knob: both public
endpoints must lie within the existing per-image trust radii of the new anchor,
the reprojected (\lVert s\rVert) must exceed `secant_norm_floor`, and the pair
must pass the existing cosine and block-noise curvature gates.  This is the
structural condition that the normal coordinate is retained and resolvable.

Takeover is global rather than scoped to one frozen subproblem.  Once latched,
all later outers forbid H0/cold selection.  A new frame uses L-BFGS only with at
least two reconstructed pairs.  Otherwise it uses the latest valid scalar

\[
\gamma=\frac{s^{\mathsf T}y}{y^{\mathsf T}y},
\]

clipped to the already fixed interval ([0.25\alpha,4\alpha]), with its source
state persisted.  This is labelled a warm scalar safeguard, not current-frame
L-BFGS memory.  An inadmissible safeguard causes HOLD only when uncertainty is
reducible; otherwise the run terminates without cold recovery.

## Public state and NoF requirements

The minimum explicit phases are `ANCHORING`, `READY`, `MOVE_PREPARED`,
`RESAMPLE_PREPARED`, `HOLD`, and `TERMINAL`/`FAILED`.  Every complete band is
published immediately.  A complete band has exactly 20 internal images and
costs exactly 20 mean-force evaluations; both force components at one 2D
collective-variable point jointly count as one evaluation.  No unselected path
may be evaluated.  Interrupted image evaluations remain in the absolute ledger
but do not form a public band.  Resume must neither hide nor double-count them.

Prepared and `FAILED` phases are durable in status.  A zero-NoF `TERMINAL` is
durable in status/completion; resume validates the public trajectory and
returns that terminal result without regenerating a proposal or calling the
oracle.

An outer-boundary refresh costs zero NoF but is an atomic checkpoint because it
changes \(A,B,G,z,P\), the eligible secants, and the next direction.  Therefore
both \(r\) and \(m_{\max}\) must cause observable trajectory branching in
engineered tests; they cannot remain labels on identical trajectories.

The historical shared initial observation remains the common causal input at
incremental NoF zero.  It lacks the original ten block means, so it is handled
by one explicit `initial_no_se_special_case`: SE is marked *unavailable*, never
set to a falsely precise zero; no noise-resolved stopping or curvature claim is
made; and exactly one frozen initial-scale proposal is article-reparameterized
and evaluated as the first 20-NoF public state.  From state 1 onward, every
noise gate and every hold/resample decision requires genuine block estimates.
This preserves the same state-0-to-state-1 opportunity as current IVM instead
of charging V14's first checkpoint only for a same-path bootstrap.

## Systematic development mesh

The first full-factorial screen is

\[
\alpha\in\{10,13,16\}\times10^{-5},\qquad
r\in\{0.4,0.7\},\qquad
m_{\max}\in\{4,8\}.
\]

This gives 12 combinations.  The integer mantissas bracket the first-round
\(13\times10^{-5}\) value symmetrically; the two residual reductions and two
internal-step caps are deliberately separated enough to reveal whether the
new controls change the trajectory.  All other controls are held fixed.
Every combination is replayed under both the deterministic FES oracle and the
same fixed synthetic-noise seeds.  Any refinement must be a complete,
predeclared bracket or factorial refinement, never one attractive decimal
inserted after inspecting results.

## Required tests before new restrained MD

1. Frozen basis, periodic coordinate, and gauge identities.
2. Actual post-reparameterization \(z\) and \(\delta\), including an example
   where nominal and actual steps differ.
3. The \(P\) and \(y\) equations, with no duplicated \(s/\alpha\).
4. Ten-block standard error and weighted pooling against analytic examples.
5. Forced low-noise movement and high-noise hold/resample branches.
6. Resampling preserves path bytes/hash/certificate, adds 20 NoF, and does not
   increment the movement counter or create \(s=0\) memory.
7. One-way warm takeover and absence of cold recovery afterward.
8. Engineered \(r\)-stop and \(m_{\max}\)-stop cases yield different next paths.
9. One complete band equals 20 NoF; partial-band ledger, resume, and
   budget-prefix invariance are explicit.
10. No hidden oracle calls and no evaluation of unselected candidates.
