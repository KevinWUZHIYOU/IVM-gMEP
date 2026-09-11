# Chignolin v21 bicriteria IVM production report

## Status and preserved predecessor

This report freezes the equidistance-improved v21 production result. The
previous high-accuracy v5 result, its report, and its source archive remain
unchanged under `snapshots/state_policy_v5/`. v5 remains a useful reference
when terminal `I_perp` alone is prioritized; v21 is the selected
three-criterion Pareto result when path parameterization must also remain
well resolved.

The production trajectory is a single 500-call run. It is not assembled from
screening runs. OpenMM used the CPU platform in double precision with one CPU
thread. An independent run with the same frozen configuration reproduced
every stored array exactly, excluding only the run label and wall-clock time.

## IVM theory status

Every outer iteration solves the frozen-normal proximal IVM subproblem

\[
\min_{\delta_i}
U(x_i+P_i\delta_i)+\frac{1}{2\alpha}\lVert P_i\delta_i\rVert^2,
\qquad
P_i=I-\tau_i\tau_i^\mathsf T,
\qquad
\tau_i=\frac{x_{i+1}-x_{i-1}}
{\lVert x_{i+1}-x_{i-1}\rVert}.
\]

The tangent and projector are frozen while each image subproblem is solved.
Projected L-BFGS and Armijo backtracking are numerical solvers for this
subproblem; they do not replace it. Curvature history is accumulated inside a
fixed-normal-space subproblem and is not transferred blindly across rotating
normal spaces.

Adaptive alpha, refreshed path-level globalization, and optional spring
reparameterization are practical safeguards around the same IVM
subminimization. They are compatible with the projected proximal
interpretation, but the optional reparameterization and adaptive-alpha policy
are outside the manuscript's ideal fixed-alpha, unreparameterized convergence
theorem.

## Transferable state policy

The production algorithm has no call-index schedule.

- Alpha starts at 1 and is bounded by `[0.03, 20]`. Actual/predicted proximal
  decrease below `0.25` shrinks alpha by `0.5`; agreement above `0.75` grows
  it by `1.5`.
- The atom-wise trust radius is twice the median adjacent-image RMS
  displacement, with a 1 Angstrom cap. It therefore scales with molecular
  size and image resolution.
- A spring proposal becomes eligible only after
  `I_perp/I_perp(initial) <= 1e-6`, physical normal
  `fmax <= 0.006 eV/Angstrom`, and segment-length CV exceeds `0.10`.
  The `0.006` threshold is 12% of the declared `0.05 eV/Angstrom`
  physical-force convergence standard.
- Reparameterization uses only the standard adjacent-length-difference spring
  force along the shared central-difference tangent. The geometry-only
  proposal removes 25% of the current CV error and calls neither OpenMM energy
  nor force.
- The spring macrostep is committed only after real frozen-normal IVM
  subminimization restores the preproposal `I_perp` and improves CV by at
  least 5%. Rejected and rollback evaluations remain in the force-call
  counters.
- After commit, the raw requested equidistance integral may not exceed
  `1.02` times its committed value.
- A synchronized frozen-normal update is tried first. If it exceeds the
  equidistance ceiling, a bicriteria merge searches already-evaluated
  contiguous image blocks. A block is accepted only if refreshed
  central-tangent `I_perp` does not increase and the raw integral stays below
  the global ceiling. This selection makes no extra OpenMM call.

The last item replaces the previous all-or-nothing rollback. It is a
path-level globalization rule: the candidate coordinates, energies, and
forces still come exclusively from genuine IVM subproblem evaluations.

## Fixed inputs and production files

- IVM production:
  `ivm_state_policy_v21_bicriteria_outer_production_central_state_ivm_cfg4c72e6afade7_budget500.npz`
- Dense IVM diagnostics:
  `dense_barrier_ivm_state_policy_v21_bicriteria_outer_production.npz`
- Best fixed NEB:
  `neb_production_central_k20_alpha0p0001_ode_budget5000.npz`
- Initial path SHA-256:
  `d2470e4fde723f07e1053eaae9ecd1f2f639ef8ca75586798f79f38d81dbf081`
- Final v21 path SHA-256:
  `654ad4aa77a282187ce0296099a4242237e6ffd5478724bb7654d9b544288952`
- Complete stored-path array SHA-256:
  `78a84d9f4444afe72558279f2e4c49a6ac4132fb64dd56d15e64a644a969b31c`

All tangent-dependent operations use the normalized central difference.

## Unnormalized I_perp and NEB envelope

| calls/image | v21 IVM | best NEB envelope | NEB/IVM |
|---:|---:|---:|---:|
| 2 | 2.9658e4 | 7.3736e4 | 2.49 |
| 5 | 2.7194e3 | 1.9218e4 | 7.07 |
| 10 | 3.0773e2 | 3.4085e3 | 11.08 |
| 20 | 1.3854e1 | 3.5371e2 | 25.53 |
| 50 | 5.9743e-1 | 5.5561e1 | 93.00 |
| 100 | 9.6871e-2 | 1.0393e1 | 107.28 |
| 200 | 4.4873e-2 | 2.3118 | 51.52 |
| 300 | 1.5939e-2 | 1.0629 | 66.69 |
| 400 | 1.5136e-2 | 6.5111e-1 | 43.02 |
| 500 | 1.4922e-2 | 4.5474e-1 | 30.48 |

Across calls 2--500, v21 has zero `I_perp` losses and zero sparse forward
barrier losses against the pointwise envelope of all 11 NEB configurations.
The minimum, median, and call-500 ratios
`I_perp(NEB envelope) / I_perp(IVM)` are `2.49x`, `68.8x`, and `30.5x`.

At call 500, v21 has:

- unnormalized `I_perp = 0.0149217521`;
- physical normal `fmax = 0.00450567 eV/Angstrom`;
- sparse 22-image forward barrier `18.77242 kcal/mol`.

The terminal `I_perp` is 4.8% above v6 (`0.01423641`), while remaining far
below the best fixed NEB at 5000 calls (`0.02843174`).

## Adaptive dense 0 K barrier

Every one of the 500 stored production paths was evaluated by independent
diagnostic evaluators. Each adjacent segment was compared at 10 and 20
subdivisions; 44 records required 40 subdivisions, and none required 80.

- v21 IVM at call 500: `21.0517364 kcal/mol`;
- v6 IVM at call 500: `20.9120802 kcal/mol`;
- v5 IVM at call 500: `21.2205588 kcal/mol`;
- fixed NEB at call 5000: `20.9832055 kcal/mol`.

v21 differs from the 5000-call NEB endpoint by only `0.06853 kcal/mol`.
The finest absolute dense-barrier ceiling reached by both v21 and fixed NEB
is `21.0209477 kcal/mol`: v21 first reaches it at 251 calls and NEB at 2960
calls, an `11.79x` first-passage speedup.

Dense diagnostic backend calls and cache misses are both exactly `90,197`.
They are excluded from the optimization axis.

## Equidistance improvement

The monitored quantity is

\[
\int_0^1 [\phi'(s)\cdot\phi''(s)]^2\,ds,
\]

with the same centered finite differences used throughout. For equally spaced
image labels, its discrete identity is proportional to the squared
differences of adjacent squared segment lengths, so it measures
constant-speed parameterization rather than path curvature.

The spring macrostep committed at call 251:

- raw integral: `1.5660e9 -> 1.2149e9` (22.4% decrease);
- segment-length CV: `0.22840 -> 0.18189`;
- proposal fraction of the full pure-spring relaxation: `0.32598`;
- recovery cost: 413 genuine joint evaluations across the internal images;
- committed `I_perp`: `0.0177167`.

The bicriteria outer merge then reduced `I_perp` without exceeding the global
integral ceiling. At call 500:

| metric | v6 | v21 | v21 change |
|---|---:|---:|---:|
| raw integral | 1.57169e9 | 1.23912e9 | -21.2% |
| dimensionless relative defect | 1.36582 | 1.07147 | -21.6% |
| segment-length CV | 0.187585 | 0.179532 | -4.3% |
| max/min segment ratio | 1.85574 | 1.79955 | -3.0% |

NEB remains almost perfectly equidistant by construction
(`raw integral ≈ 2.40e2`, `CV ≈ 3.80e-4`). v21 does not claim to match that
limit; it materially narrows the gap while keeping IVM's force and
barrier advantages.

## Rejected alternatives kept for audit

- A 50% CV contraction reached a raw integral near `1.31e9`, but its dense
  barrier rose to `35.66 kcal/mol`.
- A 65% CV contraction produced a dense barrier of `36.25 kcal/mol`.
- An early `fmax <= 0.01` trigger improved CV to about `0.162`, but the normal
  solve stalled at `I_perp ≈ 0.0636`.
- Direct aggressive raw-integral targets failed the simultaneous recovery
  test and were rolled back.

These negative results motivated the conservative 25% CV correction, the
state trigger, and the independent dense-barrier veto. They are not spliced
into the production trajectory.

## Joint evaluation and reproducibility audit

- Internal-image min/mean/max/sum: `259 / 392.50 / 500 / 7850`.
- All 22 image evaluators: `7852` real joint evaluations.
- Raw optimization OpenMM backend: `7852` joint evaluations.
- Energy API requests: `8210`; force API requests: `8210`.
- Current-run coordinate-cache hits: `358`.
- Backend calls equal image cache misses exactly; the audit flag is true.
- The single spring attempt, all recovery evaluations, rejected candidates,
  and rollbacks remain counted.
- Diagnostic calls are reported separately and do not enter the horizontal
  axis.

The independent screen and production runs have identical final-path and
stored-path hashes and no array mismatch apart from run label and wall time.
All 46 pure logic regressions pass, including the collective central-tangent
coupling case for bicriteria merging.
