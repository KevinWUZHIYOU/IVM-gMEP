# Chignolin state-driven IVM production report

## Theory status

The production algorithm solves the discrete frozen-normal proximal IVM
subproblem at every outer iteration:

\[
\min_{\delta_i}
U(x_i+P_i\delta_i)+\frac{1}{2\alpha}\lVert P_i\delta_i\rVert^2,
\qquad
P_i=I-\tau_i\tau_i^\mathsf T,
\qquad
\tau_i=\frac{x_{i+1}-x_{i-1}}
{\lVert x_{i+1}-x_{i-1}\rVert}.
\]

This is the Euclidean discrete specialization of the projected proximal
functional and independent image subproblems in the manuscript. Tangents,
projectors, and node geometry are frozen while an inner subproblem is solved.
Every accepted displacement remains in that frozen normal space.

L-BFGS, Armijo line search, and the force-field Hessian preconditioner are
inner numerical solvers for this same objective; they do not replace the IVM
subproblem. The preconditioner is constructed without an OpenMM evaluation.
L-BFGS history is retained inside a fixed-tangent subproblem, but is not reused
across outer subproblems by default. A controlled continuation test showed
that outer-history reuse worsened the terminal unnormalized `I_perp` from
approximately `0.00832` to `0.00899`, because the central-difference normal
spaces rotate between outer iterations.

Adaptive alpha, refreshed-`I_perp` globalization, and optional spring
reparameterization are practical safeguards. They are compatible with the
projected proximal objective but are not covered by the manuscript's ideal
fixed-alpha, unreparameterized convergence theorem.

## Transferable state policy

The selected production run contains no force-call-index schedule.

- Alpha starts at 1 and is restricted to `[0.03, 20]`.
- The actual/predicted proximal decrease ratio controls alpha. Ratios below
  `0.25` shrink alpha by `0.5`; ratios above `0.75` grow it by `1.5`. These are
  standard trust-region quality thresholds, not Chignolin call stages.
- The atom-wise trust radius is twice the median RMS displacement between
  adjacent images, with a 1 Angstrom safety cap. It therefore scales with
  molecular size and image resolution.
- The inner solve stops at a relative KKT residual of `0.1` or its local
  evaluation allowance. Armijo backtracking protects the exact proximal
  objective.
- Once physical normal `fmax <= 0.05 eV/Angstrom`, completed frozen-tangent
  subproblem solutions are committed collectively. At
  `fmax <= 0.005 eV/Angstrom`, refreshed central-tangent `I_perp` is protected
  strictly. These state changes do not alter alpha or the trust radius.
- A geometry-only spring proposal is eligible only at
  `fmax <= 0.005 eV/Angstrom` and segment-length CV above `0.10`. It removes
  25% of the current CV error. It is committed only after real IVM
  subminimization restores the preproposal `I_perp` and improves CV by at
  least 5%.
- After a successful spring macrostep, hysteresis prevents another attempt
  until CV deteriorates back to the level that triggered the preceding
  correction. No force-call or total-budget value enters this rule.

## Fixed runs and initial path

- IVM:
  `ivm_state_policy_v5_hysteresis_production_central_state_ivm_cfg580da24ccdfc_budget500.npz`
- Best fixed NEB:
  `neb_production_central_k20_alpha0p0001_ode_budget5000.npz`
- Both initial paths have SHA-256
  `d2470e4fde723f07e1053eaae9ecd1f2f639ef8ca75586798f79f38d81dbf081`.
- All tangent-dependent operations use normalized central differences.

## Unnormalized I_perp

| calls/image | state IVM | old tuned IVM | best NEB envelope |
|---:|---:|---:|---:|
| 2 | 2.9658e4 | 3.8868e4 | 7.3736e4 |
| 5 | 2.7194e3 | 7.6970e3 | 1.9218e4 |
| 10 | 3.0773e2 | 2.1260e3 | 3.4085e3 |
| 20 | 1.3854e1 | 2.4261e2 | 3.5371e2 |
| 50 | 5.9750e-1 | 1.4795e1 | 5.5561e1 |
| 100 | 9.6526e-2 | 2.4362 | 1.0393e1 |
| 200 | 3.6403e-2 | 5.4096e-2 | 2.3118 |
| 300 | 1.2452e-2 | 2.6353e-2 | 1.0629 |
| 400 | 9.0408e-3 | 1.0752e-2 | 6.5111e-1 |
| 500 | 8.1518e-3 | 8.8562e-3 | 4.5474e-1 |

The state IVM has zero losses against the 11-configuration pointwise NEB
envelope over calls 2--500. The minimum, median, and call-500 ratios
`I_perp(NEB envelope) / I_perp(IVM)` are `2.49x`, `76.9x`, and `55.8x`.

The finest common `I_perp` target with the extended fixed NEB is
`0.0284317`. IVM first reaches it at 211 calls and NEB at 5000 calls, a
`23.70x` first-passage speedup. NEB does not reach the IVM@500 value within
5000 calls.

At call 500, physical normal `fmax` is `0.003538 eV/Angstrom`, compared with
`0.003821 eV/Angstrom` for the old tuned IVM.

## Adaptive dense 0 K barrier

All 500 stored IVM paths were postprocessed by separate diagnostic evaluators.
Each segment was first subdivided 10 and 20 times; 44 records required 40
subdivisions and none required 80. Diagnostic backend calls and evaluator
cache misses are both exactly 99,101 and are excluded from the optimization
axis.

- State IVM@500: `21.22056 kcal/mol`
- Old tuned IVM@500: `22.69327 kcal/mol`
- Extended NEB@5000: `20.98321 kcal/mol`
- Lowest extended-NEB record: `20.39540 kcal/mol`

Relative to the extended NEB endpoint, the IVM barrier discrepancy falls from
`1.71006` to `0.23735 kcal/mol`, a `7.20x` reduction. Relative to the lowest
observed extended-NEB value, the endpoint error falls from `2.29787` to
`0.82516 kcal/mol`.

The finest common absolute forward-barrier ceiling is
`20.83376 kcal/mol`. IVM first reaches it at 265 calls and NEB at 3060 calls,
an `11.55x` first-passage speedup.

## Equidistance audit

The spring macrostep committed at call 263:

- segment-length CV: `0.23094 -> 0.18270`;
- max/min ratio was reduced at the same macrostate;
- raw requested integral:
  `1.7897e9 -> 1.3038e9`;
- `I_perp`: `0.01781 -> 0.01656`.

Subsequent normal IVM updates gradually distorted the parameterization. At
call 500:

- CV is `0.21263`, better than the old IVM's `0.24162`;
- max/min is `2.1018`, better than the old IVM's `2.3092`;
- the raw integral is `4.2802e9`, 10.0% worse than the old IVM's `3.8916e9`.

Thus equidistance is improved in the directly interpretable global spacing
statistics, but not in the high-frequency-sensitive raw integral. A diagnostic
3% late spring correction failed to recover the preceding `I_perp` within the
same 65-call remainder and was not selected. This tradeoff must remain visible
rather than being hidden by another tuned late stage.

## Joint evaluation audit

- Internal-image min/mean/max/sum: `304 / 433.55 / 500 / 8671`.
- All 22 image evaluators: `8673` real joint evaluations.
- Raw optimization OpenMM backend: `8673` joint evaluations.
- Equality is exact and the audit flag is true.
- The single delayed spring attempt and every recovery/rejected evaluation
  remain in these counters.
- Returning to a coordinate evaluated earlier in the same run reuses that
  run-local energy/force record exactly; histories are never shared between
  experiments.
