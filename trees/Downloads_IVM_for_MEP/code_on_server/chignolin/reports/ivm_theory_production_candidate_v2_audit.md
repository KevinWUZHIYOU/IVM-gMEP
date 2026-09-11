# Chignolin IVM production candidate v2 audit

## Declared comparison

- IVM source: `ivm_production_central_hybrid_cfg4c8e4f96a7ee_budget500.npz`
- IVM comparison budget: 500 maximum joint evaluations per internal image.
- Fixed NEB source: `neb_production_central_k20_alpha0p0001_ode_budget5000.npz`.
- NEB optimizer: ODE with `k=20`; no FIRE and no CI.
- Both methods use the same 22-image IDPP initial path and the same normalized
  central-difference tangent, `x[i+1] - x[i-1]`, everywhere.
- The displayed IVM curve is one independent, strict 500-call production run.
  It is not stitched from tuning runs.

## IVM subproblem

For each frozen central-difference tangent, every internal image solves

\[
\min_{\delta\perp\tau_i}
V(x_i+\delta)+\frac{\|\delta\|_2^2}{2\alpha}.
\]

The returned gradient is the projected physical gradient plus
`delta / alpha`. The early projected L-BFGS iterations and the subsequent
per-image L-BFGS-B solves use this same IVP objective. The optional acceptance
guard is applied only after a real candidate evaluation; rejected and rollback
evaluations remain counted.

## Main results at the 500-call declaration

| Quantity | IVM | NEB at 500 | Fixed-NEB first passage | Speedup |
|---|---:|---:|---:|---:|
| Unnormalized `I_perp` | 0.09036025 | 0.45866436 | 1723 calls | 3.446x |
| Adaptive dense barrier error | 3.20897113 kcal/mol | 5.34490023 kcal/mol | 1260 calls | 2.520x |

- The new `I_perp@500` is 7.88% below the previous selected IVM value,
  0.09808569.
- With a relative comparison tolerance of `1e-9`, the new IVM `I_perp` is
  never worse than the fixed NEB at any recorded call from 1 through 500.
- The barrier reference is 20.98320551 kcal/mol. The IVM dense barrier at 500
  is 24.19217664 kcal/mol.
- On the common-threshold curve, IVM first reaches its final barrier-error
  threshold at call 490, giving 1260 / 490 = 2.571x. The conservative declared
  endpoint speedup uses 1260 / 500 = 2.520x.
- The updated four-condition NEB stopping point remains call 3620.

## Joint energy-force counter audit

At the declared IVM call 500:

- Per-internal-image joint evaluations:
  `[475, 497, 495, 499, 497, 499, 500, 497, 500, 500, 500, 496, 500, 495, 500, 499, 499, 498, 498, 471]`.
- Internal-image min/mean/max/sum: `471 / 495.75 / 500 / 9915`.
- Including the two endpoints: 9917 actual joint OpenMM evaluations.
- Energy API requests: 10153.
- Force API requests: 10153.
- Every request event requested both cached quantities, so the two API totals
  describe the same 10153 events rather than 20306 physical evaluations.
- Cache hits: 236.
- Identity: `9917 actual evaluations + 236 cache hits = 10153 request events`.
- Budget-blocked cache misses: 0.

For the formal 500-call production run:

- Bottom-level OpenMM joint evaluations: 9917.
- Sum of the 22 independent image-evaluator joint counters: 9917.
- Counter consistency: passed exactly.

An independent 1000-call validation run reproduced every path, metric, and
counter record in the production prefix through call 500 exactly. Its complete
1000-call backend and summed-image totals also agree exactly at 19603.

The 53-path dense-barrier postprocessing used 45268 additional diagnostic
OpenMM evaluations. Its backend total equals its diagnostic cache-miss total,
and these calls are explicitly excluded from the optimization horizontal axis.

## Automated checks

All 23 tests pass, including:

- exact one-entry cache semantics for unchanged, changed, restored, and
  one-ULP-different coordinates;
- force-to-energy, energy-to-force, repeated-property, ASE, and NEB request
  sequences without double evaluation;
- rejected and rollback evaluations remaining counted;
- full checkpoint counter/cache restoration;
- a direct L-BFGS-B IVP suboptimization counter audit;
- exact backend-versus-image counter equality;
- central-difference tangent and unnormalized `I_perp` checks.
