# Chignolin fixed-\(\alpha\) mechanism audit

Date: 2026-07-28

## Question

Why does the Chignolin fixed-\(\alpha\) sensitivity curve remain a broad
right-hand plateau up to \(\alpha=10^3\), instead of forming a steep
right-hand wall? Is this caused by a hidden cap or early stopping, and should
such safeguards be removed?

## Main conclusion

The existing curve is not evidence that the nominal \(\alpha\) was silently
reduced. Every fixed-\(\alpha\) run used its declared value throughout. The
large-\(\alpha\) runs reached the full 500-call budget rather than stopping
early. The modest right-hand plateau has two distinct causes:

1. \(\alpha\) is the parameter of an implicitly solved proximal
   subminimization, not the step size of an explicit Euler update. Therefore
   large \(\alpha\) is not mathematically required to diverge.
2. The implementation accepts only certified proximal-objective proposals
   and protects the published outer path by a refreshed-\(I_\perp\)
   globalization test. At large \(\alpha\), most attempted force evaluations
   are rejected or backtracked. The guard converts nonmonotone large-\(\alpha\)
   behavior into stagnation rather than an exploding residual.

It would therefore be misleading to remove safeguards solely to manufacture
a U-shaped hyperparameter curve. The defensible presentation is to retain the
protected sensitivity plot as the robustness result and report the
accepted-state yield/backtracking diagnostics, with a matched mechanism
ablation in the supplement.

## Why the explicit-step intuition does not apply

For one frozen normal slice, IVM minimizes

\[
  U(\varphi_k+n)+\frac{1}{2\alpha}\lVert n\rVert^2,
  \qquad n\in N_{\varphi_k}.
\]

An exact stationary point satisfies

\[
  P_k\nabla U(\varphi_{k+1})+\frac{1}{\alpha}n_k=0,
  \qquad
  \varphi_{k+1}=\varphi_k+n_k.
\]

Equivalently,

\[
  \varphi_{k+1}
  =\varphi_k-\alpha P_k\nabla U(\varphi_{k+1}),
\]

which is an implicit/backward-Euler or proximal relation. On a positive
quadratic mode with curvature \(\lambda>0\), the update factor is
\(1/(1+\alpha\lambda)\), not \(1-\alpha\lambda\). It is stable for every
\(\alpha>0\). Large \(\alpha\) weakens the quadratic regularization and can
make a nonconvex normal-slice subproblem harder, less local, or more prone to
basin changes, but divergence is not a theoretical necessity.

The exact-objective line search is part of solving the stated proximal
subproblem reliably. Removing it would no longer be a clean test of the same
IVM subminimization.

## Audit of the protected fixed-\(\alpha\) sweep

All runs use the same initial path and use \(\alpha\) exactly as declared.
There is no call-index schedule, adaptive shrinkage, strict-polish phase, or
reparameterization attempt in this sweep.

| \(\alpha\) | final call/image | stop | final \(I_\perp\) | records with any accepted image | backtracking events |
|---:|---:|:---|---:|---:|---:|
| 0.03 | 500 | budget | 0.575406 | 100.0% | 0 |
| 0.1 | 500 | budget | 0.148358 | 100.0% | 0 |
| 0.3 | 346 | no progress | 0.087586 | 95.4% | 0 |
| 1 | 342 | no progress | 0.038852 | 94.7% | 41 |
| 3 | 347 | no progress | 0.124658 | 20.5% | 30,580 |
| 10 | 500 | budget | 0.114552 | 14.0% | 109,080 |
| 20 | 500 | budget | 0.084712 | 15.0% | 113,820 |
| 30 | 500 | budget | 0.083705 | 18.8% | 117,640 |
| 100 | 500 | budget | 0.125914 | 19.2% | 123,640 |
| 300 | 500 | budget | 0.104670 | 16.6% | 121,800 |
| 1000 | 500 | budget | 0.098739 | 16.4% | 121,200 |

The configured per-atom maximum step is \(1\) Å, while the active
geometry-derived trust radius is approximately \(0.74\)--\(0.78\) Å. This is
not an \(\alpha\) cap: the stored active-\(\alpha\) arrays contain only the
nominal values. At the common 300-call checkpoint, the accepted-state yield
falls from 100% for \(\alpha\le1\) to about 19--27% for
\(\alpha\ge3\), and roughly 98% of large-\(\alpha\) records enter
backtracking. Thus the large-\(\alpha\) regime is already computationally
poor even though the protected physical metrics do not explode.

## Matched mechanism ablations

All ablations use the same initial path, central-difference tangents, fixed
\(\alpha\), solver, inner-call limit, and force-count accounting. Each pair
was run for 150 calls per internal image
(\(N_F=3002\)). Production results were not overwritten.

Definitions:

- **Production protection:** normal trust cap plus exact proximal-objective
  line search plus refreshed-\(I_\perp\) outer globalization.
- **Relaxed trust:** increase max step from 1 to 10 Å and the trust fraction
  from 2 to 20, while retaining the outer \(I_\perp\) guard.
- **Proximal only:** retain the production trust cap and exact proximal
  objective, but remove the outer \(I_\perp\) monotonicity guard.
- **Both relaxed:** combine relaxed trust with proximal-only outer
  acceptance.

| variant | \(\alpha\) | \(I_\perp\) at \(N_F=3002\) | \(f_{\max}\) (eV Å\(^{-1}\)) | densified barrier (kcal mol\(^{-1}\)) |
|:---|---:|---:|---:|---:|
| production | 1 | 0.096807 | 0.018040 | 22.7272 |
| production | 1000 | 0.348786 | 0.021194 | 25.4410 |
| relaxed trust | 1 | 0.097388 | 0.018194 | 22.7744 |
| relaxed trust | 1000 | 0.350681 | 0.029990 | 25.4399 |
| proximal only | 1 | 0.096807 | 0.018040 | 22.7272 |
| proximal only | 1000 | 8.999404 | 0.416632 | 20.3921 |
| both relaxed | 1 | 0.097388 | 0.018194 | 22.7744 |
| both relaxed | 1000 | 1.546051 | 0.148375 | 20.7287 |

Relaxing the step/trust cap changes the protected
\(\alpha=1000\) residual by only 0.54%, so the cap is not the explanation for
the plateau. By contrast, removing the refreshed-\(I_\perp\) globalization
causes \(\alpha=1000\) to reach a transient minimum and then rebound by
33.1-fold by call 150. The final proximal-only residual is 93.0 times the
\(\alpha=1\) result. The selected \(\alpha=1\) trajectory is identical with
and without the outer guard through call 150.

The selected \(\alpha=1\) proximal-only ablation was subsequently extended
to the full 500-call budget. It stopped at call 342 for the same no-progress
condition as production. Apart from the run label, configuration hash, and
wall-clock timestamps, every common stored field is identical (with NaNs
treated as equal), including every path checkpoint, counter vector,
\(I_\perp\), barrier, \(f_{\max}\), accepted fraction, and final path. Its
backend and image-counter totals are both 6,823. Thus the outer
\(I_\perp\) guard is inactive for the selected \(\alpha=1\) trajectory and
does not create its reported performance.

The unguarded \(\alpha=1000\) path can show a lower barrier while its
\(I_\perp\) and \(f_{\max}\) are catastrophically worse. This is not improved
MEP convergence; it is a transient off-MEP path. Barrier alone must not be
used as the instability diagnostic.

## Recommended paper treatment

1. Keep the current protected fixed-\(\alpha\) sensitivity plot. Describe the
   right side as a safeguarded stagnation plateau, not as evidence that large
   \(\alpha\) is equally efficient.
2. Do not remove the exact proximal-objective line search. It is a solver
   globalization for the stated subminimization, not an ad hoc device that
   changes \(\alpha\).
3. Retain \(\alpha=1\) as the selected Chignolin value because it gives the
   best physical accuracy at the reported cost and a high productive-call
   yield. Do not justify it by claiming that every larger \(\alpha\) must
   diverge.
4. Add the accepted-state yield/backtracking curve, or use the mechanism
   ablation figure in the supplement. This exposes the real large-\(\alpha\)
   cost that the endpoint residual alone hides.
5. If a one-sentence summary is needed:

   > Because IVM solves an implicit proximal subproblem, large \(\alpha\) need
   > not exhibit explicit-step divergence. In Chignolin, values above
   > \(\alpha\approx3\) instead collapse the accepted-state yield and trigger
   > extensive backtracking; the outer residual safeguard converts this
   > nonmonotonic behavior into stagnation.

## Artifacts

- Mechanism figure:
  `figures/alpha_mechanism_20260728a/chignolin_ivm_alpha_mechanism_ablation.pdf`
- Figure data:
  `figures/alpha_mechanism_20260728a/data/protected_solver_mechanism.csv`
  and
  `figures/alpha_mechanism_20260728a/data/matched_ablation_endpoints.csv`
- Ablation manifest:
  `reports/ivm_alpha_mechanism_ablations_20260728a/manifest.json`
- Independent dense-barrier manifest:
  `reports/ivm_alpha_mechanism_dense_20260728a/manifest.json`
- Full selected-\(\alpha\) counterfactual:
  `runs/ivm/ivm_alpha_mechanism_proximal_only_a1_full_central_state_ivm_cfg85ee8e3b477d_budget500.npz`
