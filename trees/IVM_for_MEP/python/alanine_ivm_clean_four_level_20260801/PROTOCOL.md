# Frozen clean-start protocol

## Scientific question

Separate the effects of proximal strength, subproblem accuracy, resampling,
and production optimizations for alanine. Select a fixed integer alpha only
after every member of a layer has run with the same declared algorithm and
budget.

## Common state and independence

- Every run starts from `inputs/common_initial_path_equidistant.npz`, SHA-256
  `cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e`.
- Each `(stage, r, alpha)` member runs in a fresh process.
- Coordinates, forces, cache entries, quasi-Newton pairs, and stopping
  decisions are never shared across alpha values.
- Stage-4 history is run-local. Stored secant pairs are projected into the
  current frozen normal space before reuse.
- Endpoints, force field, image count, tangent convention, initial path, and
  output metrics are fixed.

## Four levels

### Stage 1: exact-like IVM, no resampling

Fixed alpha, proximal-only globalization, frozen-normal projected L-BFGS, up
to 80 inner iterations, and relative KKT tolerances `r=0.2` and `r=0.1`.
Reparameterization is disabled.

*Exact-like* is operational: every alpha receives the same high inner cap and
reported KKT tolerance. It does not assert an analytic exact solution when a
finite nonlinear solve reaches its cap. Residuals, unconverged-image counts,
and cap hits remain in the NPZ and audit records.

### Stage 2: exact-like IVM plus resampling

The Stage-1 subproblem policy is retained, with geometric equal-arclength
reparameterization after each outer step. Subsequent OpenMM evaluations are
charged normally.

### Stage 3: baseline inexact IVM plus resampling

Five inner iterations, `r=0.8`, proximal-only globalization, and the same
geometric resampling schedule for all alpha values. This is the deliberately
plain inexact baseline.

### Stage 4 dynamics: uniformly optimized inexact IVM plus resampling

Every alpha uses the same two-phase state machine. Phase A uses base
`r=0.96`, a 15-inner-step cap, and the fixed projected path-sum policy through
34 accepted calls; the uniformly enabled stall-tightening target is `0.95`.
Phase B starts from that accepted path, clears optimizer memory, uses `r=0.8`,
a 15-inner-step cap, precision curve reparameterization/recovery, and at most
220 calls in the selection grid. The long common-cost alpha audit preserves
the same policy and extends only the Phase-B budget. The same scalar alpha is
fixed in both phases; repeated Phase-B initialization is charged.

All image updates solve the declared fixed-alpha proximal subproblem in the
frozen normal space. Reparameterization is a separate path-coordinate
operation, and a reparameterized state must be completely re-evaluated before
it is eligible for retention. These rules are implemented by
`src/run_stage4_uniform_optimized.py` and
`src/run_ivm_v9_spring_projected_production_20260730.py`.

### Stage 4 publication archive: uniform v4 Pareto rule

The v4 publication trace applies one identical, deterministic rule to all ten
completed Stage-4 runs:

1. Activate only after raw `I_perp <= 0.007`.
2. Admit only complete Phase-A projected states or completely evaluated
   Phase-B `reparam_committed` states.
3. Screen each candidate with a 10-subdivision densified path. Every real
   OpenMM energy/force evaluation is recorded and charged as a force
   equivalent.
4. Retain a candidate only when neither raw `I_perp` nor the densified barrier
   value increases relative to the incumbent.
5. Stop the published prefix once the accepted incumbent reaches
   `I_perp <= 1.9e-4`; otherwise exhaust the available run.

No reference barrier or error-to-reference is used by this rule. The
20-subdivision reference-barrier error is recomputed only afterward for
independent reporting. Screening does not change the IVM equations or the raw
dynamics; it is an accepted-iterate/output globalization layer. The archive is
materialized by deterministic reference-free online-equivalent prefix replay.
An earlier independent online rerun verifies prefix determinism; this wording
does not claim that screening was executed inside the original optimization
process.

Any change to the dynamics or archive policy requires a new version label and
a rerun of the complete alpha grid. Alpha-specific tricks are forbidden.

## Work and metric accounting

- Optimization work is path-summed internal-image force evaluations,
  including rejected trials, line searches, recovery, and continuation
  initialization.
- Publication work adds every Pareto-screen energy/force evaluation as an
  exactly counted force equivalent.
- Per-image counters and `total_internal_joint_evaluations` are stored;
  aggregation is checked for monotonicity and backend/cache consistency.
- `I_perp` is retained raw. First-passage and accepted-incumbent curves are
  labeled separately; a cumulative minimum is never presented as raw data.
- The energy barrier is independently recomputed on densified paths. Sparse
  optimization barriers are diagnostic only.
- Missing dense data, non-monotone counters, policy mismatch, failed jobs, or
  unequal alpha policies fail closed.

## Acceptance rule and scope

The preferred alpha must lie on the common grid, use the unchanged policy,
and be evaluated jointly on early/middle/late cost, raw and accepted stability,
terminal `I_perp`, independent densified barrier accuracy, spacing, and
fixed-alpha IVM conformity. Competitor curves use frozen independently selected
best configurations.

The v4 verdict is specifically for the **published complete-state Pareto
incumbent trace**. Raw trial and complete candidates remain separately
available. The successful trace has zero rises in both reported metrics; the
raw candidate set does not, and its four barrier-error rises are explicitly
part of the acceptance evidence.
