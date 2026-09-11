# LEPS gMAM and StringNET comparison

User-requested image-chain implementations, with the existing 20 movable
images, two fixed endpoints, and exact common ASE-IDPP initial path.
No neural network is introduced by the label StringNET in this experiment:
the label denotes the user-specified two-objective schedule on this same chain.

gMAM minimizes the 22-node trapezoidal discretization of
`S = integral |grad U(phi)| |phi'| ds`.
StringNET minimizes `I_beta = log(integral exp(beta U(phi)) |phi'| ds)/beta`
for exactly the first 50 completed optimizer updates, then minimizes the
identical gMAM objective. The exponential integral uses stable log-sum-exp.
No reference barrier or normal residual enters either optimizer.

Before any production run, register the Cartesian grid of 60 combinations:
gMAM and StringNET with beta in {0.1, 0.3, 1, 3}; L-BFGS initial scales
{0.01, 0.1, 1} or Adam learning rates {0.001, 0.01, 0.1}; and either the
existing cubic equal-arclength String Method reparameterization or the
existing improved-tangent IVM identical-spring reparameterization (no recovery).
Every completed optimizer update is followed by reparameterization.
L-BFGS uses 10 curvature pairs, Armijo 1e-4, 24 halving trials, and a maximum
image displacement of 0.2 per proposed optimizer update. Adam uses
standard (0.9, 0.999), epsilon 1e-8 and the same displacement cap.
The optimizer memory is reset at the objective switch. L-BFGS secant pairs
connect consecutive reparameterized states evaluated under the same objective.
No objective or residual gate is applied to reparameterization.

Each run has 20,000 total internal-image NoF and a 5000-update safeguard.
Stop on displacement <= 1e-14 after a complete update (only after the warmup
has completed for StringNET), a failed L-BFGS line search, numerical failure,
or the budget. Every saved path has its full energy/gradient evaluated and
charged before saving. If the next complete stage would exhaust the budget,
retain the previous complete state; incurred trial work remains charged.
Endpoints are excluded from NoF, exact-coordinate cache hits are free, and
all rejected trials and reparameterized-state evaluations are charged.
Analytic autodifferentiation supplies H grad(U)/|grad(U)| for the action
gradient. Each previously uncached internal-coordinate HVP is counted
separately; it is NOT silently converted to zero total computational cost.
Objective quadrature includes the fixed endpoint values. At exactly zero
gradient the norm's zero subgradient is used. No potential smoothing is used.

Before production: validate potential values/gradients against the independent
NumPy LEPS implementation and validate both chain-objective derivatives by
central finite differences (diagnostic test only, not tangent estimation).
All path tangents used by reparameterization and residual metrics are improved
Henkelman-Jonsson tangents. No central-difference path tangents are used.
Runs execute serially under a host-local fcntl lock. This is a local-only
campaign; no remote dispatch or synchronization bypass is involved.

After all runs: independently recompute stored states and metrics. Select
one complete, validated run per method by mean log normalized I_perp plus
mean log normalized absolute densified-barrier error at NoF = 1000, 2000,
..., 20000 using last saved state at/below budget. Floors are 1e-12 for
I_perp and 1e-6 eV for barrier error; normalization uses the shared initial
values. Tie-break by terminal action, terminal NoF, config ID.
StringNET runs that never complete all 50 warmup updates and enter gMAM
are ineligible for selection. Show beta-specific and optimizer/reparameterizer
results as well as the selected curves; retain all failed/stalled results.
The old neural gMAM comparison remains a separately identified experiment.

The final all-methods figure retains its three square panels and 20,000-NoF
window. Rename only the displayed LEPS Original String label to String Method.
Archive the previous affected figure and registration before replacing it.
