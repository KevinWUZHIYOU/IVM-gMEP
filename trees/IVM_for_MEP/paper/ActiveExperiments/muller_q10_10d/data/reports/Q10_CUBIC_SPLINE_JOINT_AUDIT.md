# Q10 Muller cubic-spline NN-IVM joint audit

## Status

The locked production configuration passes the hard theory gates, the joint
direct-5001 accuracy gate at 4000 total force calls, the final-500-force
stability gate, and the three-run reproduction gate. The optional 3000-force
stretch goal is not passed.

## Acceptance rules fixed before production reruns

1. Theory gates: fixed `alpha = 1e-3`; exactly 20 moving force samples per
   objective evaluation; NEB Henkelman-Jonsson improved tangent; unconstrained
   delta optimization; only the projected IVM proximal objective; no
   Lagrangian, mixing, trust clipping, output clipping, barrier penalty, or
   dense-certificate feedback.
2. Main gate at no more than 4000 forces: direct-5001 densified
   `I_perp <= 5` and direct-5001 densified barrier error `<= 1e-5` at the same
   checkpoint.
3. Tail stability over 3500--4000 forces: `I_perp <= 7.5` and barrier error
   `<= 5e-4` at every recorded checkpoint.
4. Reproduction: three locked reruns must each end with `I_perp <= 8` and
   barrier error `<= 1e-4`.
5. Stretch goal at no more than 3000 forces: `I_perp <= 5` and barrier error
   `<= 1e-6` at the same checkpoint.

## Locked production schedule

All three stages minimize the same unconstrained projected IVM proximal
subproblem. Only the cubic B-spline approximation space and the local spacing
used to construct the improved-tangent triplet change between outer
iterations.

| Total-force interval | Cubic B-spline width | Improved-tangent stencil | Inner L-BFGS steps |
|---:|---:|---:|---:|
| 0--3200 | 12 | 0.002 | 3 |
| 3200--3700 | 14 | 0.001 | 3 |
| 3700--4000 | 8 | 0.0001 | 3 |

The collocation rule is deterministic golden shift, with 20 moving samples.
The schedule was fixed before the three production reruns and was identical in
all reruns. The three nominal seeds therefore produce identical trajectories;
this is deterministic reproducibility, not evidence of stochastic robustness.

## Production result

| Quantity | Value |
|---|---:|
| Total force calls | 4000 |
| Direct-5001 densified `I_perp` | 4.1401243274 |
| Direct-5001 densified barrier error | 1.5818082204e-6 |
| Minimum direct-5001 `I_perp` | 4.1010652562 |
| Maximum inactive transformed coordinate | 6.3695590658e-12 |
| Tail maximum `I_perp`, 3500--4000 | 4.94982597 |
| Tail maximum barrier error, 3500--4000 | 2.88786435e-4 |

All three production reruns returned these same values and passed all theory
audits. The optional 3000-force stretch goal failed: the first stage's minimum
through 3200 forces was still `I_perp = 6.5231693219`.

`Number of Forces` is the direct sum of all 20 force streams. The 40 accepted
outer updates also use 2400 energy-only improved-tangent queries; these are
reported separately and are not force calls. The 5001-point certificates are
diagnostics only and never enter the update, stopping rule, or configuration
switches.

## Formal common-certificate comparison

The existing formal postprocessor was used for every method: coordinate-wise
not-a-knot cubic densification to 5001 points followed by the NEB improved
tangent. At each method's final checkpoint within 5000 force calls, the best
non-NN densified `I_perp` is about 509.32, versus 4.14 for cubic-spline NN-IVM,
which is approximately a 123-fold reduction. The best non-NN final barrier
error is about `1.12e-5`, versus `1.58e-6` for NN-IVM, approximately a 7-fold
reduction despite NN-IVM stopping at 4000 forces.

An important limitation is retained: several baseline barrier-error curves
have isolated transient minima below `1.58e-6`. They do not simultaneously
have competitive densified `I_perp`; their `I_perp` remains roughly two orders
of magnitude larger. Therefore the honest claim is joint same-checkpoint
dominance, not pointwise dominance of the barrier-error curve alone.

## Rejected tuning routes

- Adaptive maximum-energy anchors reduced barrier error but degraded the
  whole-path residual; the best screened result was about `I_perp = 7.82` and
  barrier error `2.03e-4`.
- A uniform-collocation refinement after the golden checkpoint did not produce
  a joint improvement.
- Deeper single-subproblem L-BFGS solves did not pass the joint gate and could
  overfit the 20 collocation points.
- Clean runs with B-spline width 15 or larger were unstable or inaccurate;
  widths 18--20 produced densified residuals as large as `1e3--1e4`.
- Reversing the continuation order preserved neither low residual nor low
  barrier error.

The formal manuscript source `main.tex` was not modified.
