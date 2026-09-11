# Chignolin early-block V1 mechanism pilot

- Date: 2026-07-29 (Asia/Hong_Kong)
- Source: `run_ivm_early_block_relock_20260729.py`
- Screen session: `chig_ivm_early_block_f4`
- Run label: `production_a1_early_block_relock_f4`
- Last preserved progress before termination: maximum calls/image 49,
  raw `I_perp = 0.3332028`, sparse monitoring barrier
  `22.4281 kcal/mol`.
- The complete log remains at
  `logs/early_block_relock_20260729/screenlog.0`.

This run is a mechanism pilot only and must not be used in the paper or the
fixed-alpha comparison. Its option
`--state-curve-direct-failure-calls 4` incorrectly capped the total number of
new evaluations in a synchronized projected subproblem. That changes
successful large-alpha solves as well as failed line searches.

V2 replaces this with a universal maximum of four *rejected line-search
trials per inner iteration*. Accepted inner iterations remain governed by the
independent 15-evaluation subproblem budget and KKT stopping criterion.
Therefore V2 isolates failed-trial patience without alpha-specific truncation.
