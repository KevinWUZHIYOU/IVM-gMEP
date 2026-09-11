# Chignolin unified V2 fixed-alpha protocol audit

## Frozen implementation

- Runner:
  `src/run_ivm_early_block_relock_v2_20260729.py`
- SHA-256:
  `55c2d7143cf406e5d730239d2aae0c4998ef55301175975a24dc49dc60122d80`
- Universal launcher:
  `logs/fixed_alpha_v2_universal_20260729/launch_v2_fixed_alpha.zsh`
- SHA-256:
  `d2f580daca657b44bbedf9b0f27797543cba5dceaf648c7efc44350c8d272862`

Every alpha uses the same code and all numerical options other than the
fixed value of alpha. The common IDPP path, OpenMM CPU/double backend,
20 internal images, central-difference tangent, joint energy-force cache,
and total Number of Forces definition are unchanged.

## Subproblem and globalization semantics

- Frozen-normal proximal objective with fixed alpha.
- Synchronized equal-spring predictor: projected L-BFGS, at most 15 new
  evaluations per image, relative KKT tolerance 0.8.
- One inner line search may test at most four rejected trial scales.
  This is not a four-evaluation subproblem cap.
- If a synchronized predictor accepts no step, it is retired by a
  state-based failure event. The method continues with a greedy
  largest-normal-force block subminimization of the same fixed-alpha
  proximal objective.
- Central-Hermite spring relocking is an optional coordinate-only
  parameterization operation. It uses no OpenMM energy or force.
- The block trust fraction is fixed at `1e-4` for every alpha and every
  force budget.
- No outer `I_perp` acceptance guard and no alpha schedule are used.

## Regression evidence

- Existing accounting/tangent/string-method suite: 53 tests.
- Added V2 tests:
  `tests/test_v2_line_search_semantics.py`
  (SHA-256
  `2fad7fb78240beaa065bfa5697ebfffc1ae87aa0ab74b70a874bde623b9f02c5`).
- The V2 solver with its default 64-trial line search is bitwise identical
  to archived V1 on the quadratic fixture.
- With four rejected trials, the fixture still completes seven accepted
  inner iterations. Hence four limits only rejected line-search trials,
  not successful KKT iterations.

## Energy-barrier rule

Sparse 22-image barriers printed during optimization are monitoring values
only. Every reported barrier metric must come from a separate diagnostic
evaluator and real OpenMM energies on linearly densified adjacent-image
segments:

1. evaluate 10 subdivisions per segment;
2. evaluate 20 subdivisions per segment;
3. if successive barriers differ by more than 0.05 kcal/mol, double to 40
   and then at most 80 subdivisions;
4. report diagnostic cache misses separately and exclude them from Number
   of Forces.

Trailing-window diagnostics use total-force budgets 1000, 3000, 6000, and
10000, the last 30% of the actually attained budget, and seven measured
paths per window.

## Pending result audit

The final report must append, for every alpha:

- result and audit SHA-256;
- initial-path SHA-256;
- final and minimum raw `I_perp`;
- total Number of Forces;
- dense barrier trailing statistics and adaptive subdivision counts;
- exact equidistance integral;
- per-image joint-evaluation vector and backend/cache-miss identity;
- state at which the synchronized predictor was retired, if any.
