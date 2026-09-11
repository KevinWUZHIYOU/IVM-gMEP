# LEPS unified IR-IVM protocol

## Method frozen before the formal grid

This campaign keeps the current IR-IVM complete-normal proximal L-BFGS
subproblem and uses the Henkelman--Jónsson improved tangent everywhere.  It
does not use a central-difference tangent.

The transferred QM10 per-inner-step displacement cap is removed.  LEPS uses
the geometry-derived total maximum-atomic-displacement trust radius already
used by the LEPS article implementation:

\[
R=\min\{2\,\mathrm{median}(\text{segment RMS length}),1\}.
\]

This is a deterministic scale rule, not a searched parameter.  The proximal
objective, complete normal space, Armijo L-BFGS solution, and the public
controls \(\alpha,r,m_{\max}\) are otherwise unchanged.  Roundoff-safe Armijo
comparison and the Ala/Chignolin `I_perp`-monotone complete-band commit rule
prevent a numerically worse assembled band from replacing the current state.
All rejected trials remain charged.

Every five completed main stages, the method applies the Ala/Chignolin
reparameterization transaction:

1. one identical-spring direct proposal using improved tangents;
2. energy and force evaluation of all 20 internal images;
3. if needed, fixed-accuracy normal recovery with KKT fraction 0.8,
   synchronized one-accepted-step traces, and at most 320 calls per image
   including the direct proposal;
4. collective-first and then contiguous-block bicriteria merge;
5. atomic commit only when both the `I_perp` and equidistance conditions hold,
   otherwise rollback while retaining every Number-of-Forces charge.

Recovery never reads the tuned main-subproblem \(r\).  No QM10 recovery
routing, stage-dependent parameter schedule, article-baseline value,
reference barrier, or densified barrier metric is available to the optimizer.

## Only three searched parameters

The formal Cartesian grid contains 48 combinations:

- \(\alpha\in\{10,100,1000\}\);
- \(r\in\{10^{-3},10^{-2},10^{-1},0.3\}\);
- \(m_{\max}\in\{4,5,6,8\}\).

The values are round powers of ten or small integers; there are no fitted
values such as 0.67.  Every run starts from the exact common article initial
path, has a global 3000-NoF ceiling, and runs serially under a host-local lock.

## Formal acceptance

The article IVM's distinct accepted paths determine, before this grid is run,
the following Number-of-Forces checks:

`66, 150, 278, 468, 666, 874, 1052, 1240, 1413, 1578, 1727, 1862, 1982, 2097, 2194, 2280, 2357, 2423, 2481, 2531, 2572`.

At each check, last-observation-carried-forward selects the latest accepted
algorithm state available under that NoF.  Direct proposals and rejected
states are not scientific states, although their force calls are charged.
The common initial path at NoF 20 is equal and is excluded from the strict
claim.

A parameter combination passes only if, at all 21 checks,

\[
I_\perp^{\mathrm{IR-IVM}}<I_\perp^{\mathrm{article\ IVM}}
\quad\text{and}\quad
|\Delta E_{500}^{\mathrm{IR-IVM}}-\Delta E_{\mathrm{ref}}|
<|\Delta E_{500}^{\mathrm{article\ IVM}}-\Delta E_{\mathrm{ref}}|.
\]

`I_perp` uses the article path-length quadrature and improved tangent.  The
energy barrier uses coordinate-wise linear densification to 500 points.  The
formal winner is chosen first among strict passers by the smallest worst
two-metric ratio, then mean ratio, terminal NoF, and the three controls.

The article comparison figure may be replaced only after the complete grid,
independent metric calculation, method-contract audit, and NoF ledger audit
all pass.
