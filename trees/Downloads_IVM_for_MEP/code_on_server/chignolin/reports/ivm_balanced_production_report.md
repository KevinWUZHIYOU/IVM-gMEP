# Chignolin IVM balanced production report

## Selected fixed production runs

- IVM: `ivm_balanced_production_central_hybrid_ff_cfg03a295eee0fe_budget500.npz`
- NEB: `neb_production_central_k20_alpha0p0001_ode_budget5000.npz`
- Both runs use exactly the same 22-image IDPP initial path:
  `d2470e4fde723f07e1053eaae9ecd1f2f639ef8ca75586798f79f38d81dbf081`.
- The maximum absolute elementwise difference between the two stored initial
  paths is exactly zero.
- Every tangent-dependent operation uses the normalized central difference
  `x[i+1] - x[i-1]`.

## Cause of the previous late-stage plateau

The frozen-tangent proximal subproblem and the diagnostic `I_perp` do not use
the same tangent during an inner solve. The subproblem correctly holds its
tangent fixed, while `I_perp` is recomputed from the newly coupled path.
Consequently, a valid subproblem step may temporarily raise the refreshed
`I_perp` while rotating the path tangent toward the MEP.

The previous implementation required refreshed `I_perp` to decrease after
every partial update. That treated `I_perp` as a per-inner-step Lyapunov
function, which it is not, and rejected the tangent-realignment moves required
for continued convergence.

The implemented fix is a delayed-commit collective proximal solve:

1. Freeze the central-difference tangent and solve the image subproblems.
2. Count and retain every real trial evaluation, including rejected trials.
3. Keep the publicly accepted global path unchanged during the inner solve.
4. Commit only the completed collective candidate and then refresh the
   central-difference tangent.
5. Once `I_perp <= 0.016`, switch back to strict monotone late polishing.

The state-based alpha policy is:

- adaptive trust-quality warm-up before the force-field preconditioned regime;
- `alpha = 20` during collective tangent realignment;
- `alpha = 10` after the late-polish trigger.

The collective regime activated at call 265 and protected late polishing at
call 355. Pure spring-force reparameterization was tested but not selected:
even a 2% geometric spring move raised `I_perp` from about 0.011 to about
0.75 on the curved path. No OpenMM force or energy was used for that test.
The selected production run therefore leaves reparameterization optional and
disabled. Its segment-length CV changes only from 0.2290 to 0.2416.

## Raw unnormalized I_perp

| calls/image | IVM | best NEB envelope | NEB / IVM |
|---:|---:|---:|---:|
| 2 | 3.8868e4 | 7.3736e4 | 1.90 |
| 5 | 7.6970e3 | 1.9218e4 | 2.50 |
| 10 | 2.1260e3 | 3.4085e3 | 1.60 |
| 20 | 2.4261e2 | 3.5371e2 | 1.46 |
| 50 | 1.4795e1 | 5.5561e1 | 3.76 |
| 100 | 2.4362 | 10.3927 | 4.27 |
| 150 | 0.31827 | 3.99176 | 12.54 |
| 200 | 0.054096 | 2.31176 | 42.73 |
| 250 | 0.030942 | 1.43607 | 46.41 |
| 300 | 0.026353 | 1.06289 | 40.33 |
| 350 | 0.017075 | 0.835961 | 48.96 |
| 400 | 0.010752 | 0.651108 | 60.56 |
| 500 | 0.0088562 | 0.454742 | 51.35 |

There are zero violations of the best-NEB `I_perp` envelope over calls
2--500. The weakest pointwise lead is 1.116x at call 26, while the median
pointwise lead is 44.21x.

Relative to the previous fixed IVM production result, the final `I_perp`
improves from 0.051926 to 0.0088562, a 5.86x reduction. Final physical normal
`fmax` is 0.003821 eV/Angstrom.

## Adaptive dense 0 K barrier

The barrier diagnostic linearly densifies each adjacent image segment,
compares 10 and 20 subdivisions, and doubles to 40 or 80 when the last change
exceeds 0.05 kcal/mol. It uses a separate diagnostic evaluator and does not
contribute to the optimization horizontal axis.

- 181 IVM path records were evaluated.
- 170 records converged at 20 subdivisions, 9 at 40, and 2 at 80.
- Diagnostic OpenMM backend calls and diagnostic evaluator cache misses are
  both exactly 78,586.
- The selected IVM dense barrier is below the fixed NEB dense barrier at all
  180 compared representative calls from 2 through 500.
- The minimum absolute IVM advantage is 3.404 kcal/mol and the median
  advantage is 6.874 kcal/mol.
- At call 500: IVM is 22.6933 kcal/mol and NEB is 26.3281 kcal/mol.

The former error-based cost plot used 20.395398 kcal/mol, the lowest
adaptively densified barrier observed in the extended fixed NEB run (call
3390), as an operational reference. That reference remains recorded for
traceability, but it is no longer used in the main cost-to-precision plot.
Changing it by truncating NEB would make the reference budget-dependent.

## Cost to precision

- Finest common `I_perp = 0.0284317`:
  IVM 294 calls, NEB 5000 calls, 17.01x.
- IVM final `I_perp = 0.0088562`:
  IVM 500 calls; NEB does not reach it by 5000 calls, so the result is
  budget-censored at greater than 10x.
- The main barrier panel now asks for the minimum calls required to reach an
  absolute forward-barrier ceiling. It requires no `Delta E*`.
- At the finest common target, 21.98281 kcal/mol, IVM needs 334 calls and NEB
  needs 2390 calls: 7.16x.

## Equidistance monitoring

The requested raw integral, segment-length coefficient of variation, and
max/min segment-length ratio are stored at every IVM record. The older fixed
NEB production file predates those fields, so the same quantities are
reconstructed exactly from its stored coordinate snapshots without any
OpenMM call.

- IVM segment-length CV: 0.2290 initially, 0.2226 at call 334, and 0.2416 at
  call 500.
- IVM max/min segment-length ratio: 1.872 initially and 2.309 at call 500.
- NEB segment-length CV: 0.00644 at call 100 and 0.00150 at call 499.
- NEB max/min segment-length ratio: 1.0187 at call 100 and 1.0042 at call 499.

Thus IVM converges in normal force without converging to a constant-speed
parameterization. This explains why its `I_perp` can continue to fall while
the linearly densified barrier stops improving. A future barrier-lowering
experiment should use a delayed-commit reparameterize-and-subminimize
macrostep; simply accepting the spring move immediately creates a large
temporary `I_perp` spike.

## Joint energy-force call audit

The selected IVM audit passes all invariants:

- internal-image min/mean/max/sum: 355 / 460.2 / 500 / 9204;
- all 22 images: 9206 joint evaluations;
- raw OpenMM backend: 9206 joint evaluations;
- 167 unchanged accepted-path intervals contain new real evaluations,
  demonstrating that rejected and rolled-back trials remain counted.

The fixed NEB run has 100000 internal-image evaluations and 100002 total
22-image/backend evaluations. Diagnostic barrier calls are reported separately.
