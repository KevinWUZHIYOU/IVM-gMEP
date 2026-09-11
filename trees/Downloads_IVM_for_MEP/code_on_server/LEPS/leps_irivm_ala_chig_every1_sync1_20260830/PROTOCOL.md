# LEPS period-one single synchronized recovery protocol

This preregistered campaign starts from the corrected LEPS-native
Ala/Chignolin-style IR-IVM and attempts a reparameterization transaction after
every completed main stage.  It retains the full improved-tangent spring direct
proposal.  Only the recovery computation changes.

If the evaluated direct proposal already satisfies the existing `I_perp` and
equidistance commit guards, it is committed without a recovery trial.
Otherwise, the direct gradients construct one complete-normal direction for
every internal image using the same proximal initial inverse and the same
LEPS-native total atomic trust radius.  A common step length is selected with
no additional force calls by maximizing

\[
-t g^T d-\frac{t^2\|d\|^2}{2\alpha},\qquad 0\le t\le1.
\]

The resulting complete band is evaluated once.  The existing collective-first
and contiguous-block bicriteria merge then either commits a path satisfying
the pre-transaction guards or atomically rolls back.  There are no further
recovery cycles or force evaluations.

Thus a completed transaction costs at most one direct evaluation and one
recovery evaluation per internal image: at most 40 joint Number of Forces for
the 20-image LEPS band.  Recovery reads `alpha`, as the original proximal
recovery does, but does not read the tuned main-subproblem `r`.  No QM10
routing, article baseline, reference barrier, densified barrier, or acceptance
checkpoint is available to the optimizer.

All other conditions are unchanged: improved tangent everywhere, the same
complete-normal proximal Armijo L-BFGS main problem, every-outer transaction
schedule, 3000-NoF global limit, charged rejected trials, and the original
clean 48-member grid:

- `alpha` in `{10, 100, 1000}`;
- `r` in `{1e-3, 1e-2, 1e-1, 0.3}`;
- `m_max` in `{4, 5, 6, 8}`.

The unchanged strict acceptance rule requires both article metrics to be
strictly better than article IVM at all 21 registered same-NoF checks.
