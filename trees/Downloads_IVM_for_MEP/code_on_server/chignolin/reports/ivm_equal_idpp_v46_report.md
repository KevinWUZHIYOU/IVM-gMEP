# Chignolin equal-IDPP IVM production report (v46)

## Outcome

The selected production run is `ivm_equal_idpp_v46_transactional_kkt080_production`.
It uses the same 22-image, IDPP-derived initial path as the fixed `k=50`
ODE-only NEB baseline.  The path hash stored by both results is
`57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996`.

At 500 joint evaluations per internal image, v46 gives:

- unnormalized \(I_\perp = 0.0534725517\);
- densified 0 K forward barrier \(=21.0265713\) kcal/mol;
- physical normal \(f_{\max}=0.0105658\) eV/Å;
- equidistance integral
  \(\int_0^1(\phi'\cdot\phi'')^2\,ds=1.57577\times10^{-19}\);
- segment-length CV \(=1.35218\times10^{-15}\).

The fixed `k=50` NEB first reaches the same \(I_\perp\) threshold at call
3063 and the same densified-barrier threshold at call 3100.  Using the exact
first-hit definitions, the speedups are therefore \(6.14\times\) and
\(6.20\times\), respectively.  Across the plotted precision grid, every
\(I_\perp\) speedup is between \(3.11\times\) and \(6.29\times\), and every
densified-barrier speedup is between \(6.00\times\) and \(6.50\times\).

## Why low \(I_\perp\) and equal spacing appeared incompatible

They are not incompatible in the continuous string formulation.  The
apparent conflict came from the 22-image discretization:

1. The central tangent
   \[
   \tau_i=\frac{x_{i+1}-x_{i-1}}{\|x_{i+1}-x_{i-1}\|}
   \]
   is not invariant to a finite reparameterization when adjacent spacings
   are unequal.
2. A path can therefore obtain a small *discrete* \(I_\perp\) partly by
   changing its parameterization.  Replacing that parameterization by
   equal-length springs rotates the finite-difference tangent and exposes
   a larger refreshed normal force.
3. Per-image acceptance after a global spring move creates alternating
   short/long segments.  The raw equidistance integral contains a
   second-difference-like term and amplifies that high-frequency spacing
   error even when the segment CV looks small.

This explains why earlier runs could reduce \(I_\perp\) or the spring
residual separately but not preserve both after a full reparameterization.
The remedy is an operator split that treats the physical subproblem and the
spring manifold as two coupled, but distinct, operations.

## Selected algorithm

Each outer iteration performs the following operations.

1. Freeze the same central-difference tangent used by all diagnostics.
2. For every internal image, solve the genuine frozen-normal proximal IVM
   subproblem
   \[
   \min_{\delta_i\perp\tau_i}
   E(x_i+\delta_i)+\frac{\|\delta_i\|^2}{2\alpha}
   \]
   with projected L-BFGS and a line search.  The fixed relative KKT forcing
   tolerance is 0.8.
3. Project the complete candidate band, not individual accepted images, to
   the identical-spring equilibrium of a central-Hermite representation.
   This operation uses only coordinates and spring geometry; it makes no
   OpenMM request.
4. Evaluate the projected band and commit it transactionally only when the
   local proximal conditions and refreshed global \(I_\perp\) non-increase
   condition hold.  Otherwise roll back the complete band.  Every rejected
   OpenMM evaluation remains counted.

The proximal controller was enabled with \(\alpha\in[0.03,20]\), but the
model-quality controller kept \(\alpha=1\) for the entire selected run.
There is therefore no call-index schedule or hand-tuned phase switch in the
production trajectory.  L-BFGS curvature is accumulated inside each fixed
normal-space subproblem.  Curvature is not carried across outer iterations,
because the central tangent rotates and old secant pairs then belong to a
different normal space.

The spring projection is not an energy/force minimization and contributes
zero optimization calls.  Its only OpenMM cost is the subsequent projected
coordinate evaluation, which is already included in the per-image counters.

## Pointwise comparison and startup granularity

Against the pointwise envelope of all completed NEB `k` scans, both v46
metrics are lower at call 3 and at every call from 6 onward.  The literal
per-call exceptions are:

- \(I_\perp\): calls 2, 4, and 5;
- sparse barrier: calls 2 and 5.

Call 2 is the first uncommitted inner evaluation of an atomic IVM
subproblem.  Calls 4–5 are the corresponding short publication plateau.
These evaluations are retained on the honest loss-versus-call curve and in
the audit; they are not hidden or relabeled.  From the first fully resolved
startup cycle onward, the lead persists through 500 calls.

## Barrier calculation

For each stored path, every adjacent image pair is linearly subdivided 10
and 20 times and evaluated by a separate OpenMM diagnostic evaluator.  The
subdivision count is doubled to 40 and then 80 only if consecutive levels
differ by more than 0.05 kcal/mol.  All six v46 snapshots selected 20
subdivisions; the largest 10-versus-20 difference was 0.02324 kcal/mol.
Diagnostic calls are excluded from the optimization horizontal axis.

## Call audit

For v46, the 20 internal-image joint-evaluation vector has
min/mean/max/sum \(=491/494.45/500/9889\).  Including the two endpoint
evaluations gives 9891 raw OpenMM calls, exactly matching the summed
per-image cache misses.

For the 5000-call NEB run, the internal vector is exactly 5000 for every
image: min/mean/max/sum \(=5000/5000/5000/100000\).  Including endpoints
gives 100002 raw OpenMM calls, again exactly matching the cache misses.

The v46 dense-barrier diagnostic used 2522 calls and the NEB grid used
38550 calls.  In both cases the diagnostic backend count equals the
diagnostic cache-miss count, and neither is mixed into the optimization
axis.

The run-local evaluator stores every byte-identical coordinate encountered
in that experiment together with its joint energy/force result.  Returning
to an earlier coordinate within the same experiment is a cache hit; caches
are never shared between experiments.

## NEB stopping status

The extended NEB was run to the hard limit of 5000 calls per internal image.
Although it passed the v46 final \(I_\perp\) and barrier thresholds, its
\(I_\perp\) changed by about 14% over the final 100-call endpoint window.
It is therefore recorded as a budget-limit termination rather than a
stable-convergence termination.

## Verification

- The final automated production validator passes without failures.
- The complete test suite passes: 47 tests.
- Production source contains no ASE improved-tangent implementation.
- Both production results declare the same central-difference tangent and
  the same elementwise initial path.

The continuous operator-splitting interpretation follows the simplified
string-method separation of normal evolution and reparameterization.  The
NEB comparison uses the standard separation of physical normal force and
tangential spring force; no improved, energy-weighted tangent is used here.

