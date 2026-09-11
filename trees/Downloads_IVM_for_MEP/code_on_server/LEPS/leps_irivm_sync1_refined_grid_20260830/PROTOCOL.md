# LEPS single synchronized recovery refined-grid protocol

This is a second-stage, fully Cartesian refinement of the preregistered LEPS
single synchronized recovery experiment.  The optimizer is frozen before all
new runs.  No force, path, article trajectory, densified barrier, or interim
selection result may change the registered grid after execution begins.

The only tuned controls and their registered values are

\[
\alpha\in\{20,50,100,200,500\},
\]

\[
r\in\{10^{-3},3\times10^{-3},10^{-2},3\times10^{-2},10^{-1}\},
\]

and

\[
m_{\max}\in\{3,4,5,6,8,10\}.
\]

All 150 combinations must run with a 3000-NoF limit and `jobs=1`.  Existing
first-stage results are not copied into the new campaign.

The algorithm is unchanged from
`leps_irivm_ala_chig_every1_sync1_20260830`: complete-normal proximal Armijo
L-BFGS main stages, improved tangents everywhere, one full spring
reparameterization after every completed outer iteration, and at most one
synchronized complete-band recovery evaluation.  Every rejected direct or
recovery proposal is charged.  Each transaction is capped at 20 direct plus
20 recovery Number of Forces and commits atomically or rolls back.  Recovery
does not read the main-subproblem `r`.

The strict acceptance rule is unchanged: excluding the exactly equal common
initial path, both article `I_perp` and the absolute 500-point densified
barrier error must be strictly below article IVM at all 21 registered same-NoF
checks.  Selection orders candidates by strict pass, then worst joint metric
ratio, mean joint ratio, terminal NoF, `alpha`, `r`, and `m_max`.
