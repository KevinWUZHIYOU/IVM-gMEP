# Stage-4 v4 acceptance summary

Verdict: **accepted**, with scope restricted to the published complete-state
Pareto incumbent trace. The machine-readable source of truth is
`stage4_v4_fail_closed_acceptance.json`.

## Selected result

- Fixed alpha: exact integer `20` in both phases.
- Selected optimization call: `2507`.
- Screening work: `2301` force-equivalent evaluations.
- Total charged work: `4808`.
- Raw `I_perp`: `1.8216907465962214e-4`.
- Independent 20-subdivision barrier: `2.4551593991015093 kcal/mol`.
- Independent barrier absolute error: `9.231140237990765e-4 kcal/mol`.
- Selected path SHA-256:
  `009b6d129797a54b02f6c7424426ea24c76367a184f83e0066bfbcb9fae0bcff`.
- ODE-NEB first-passage work at the matched `I_perp` threshold: `83522`.
- Charged cost speedup: `17.3714642263x`.

The accepted calls are `660, 680, 829, 943, 2507`. Their raw `I_perp` and
independent barrier errors are both monotonically non-increasing. The selected
state also beats the frozen ODE-NEB terminal result on both `I_perp` and
barrier error.

## What is and is not accepted

The Pareto screen is deterministic, identical for all alpha values,
reference-free, and exactly force-charged. It retains only completely evaluated
states and does not modify the fixed-alpha IVM update. The archive is
materialized through online-equivalent prefix replay of already completed
dynamics.

This is not a claim that every internal candidate is monotone. The retained
raw candidate diagnostic contains four densified-barrier-error rises and a
maximum shoulder of `0.2015568082 kcal/mol`. Those candidates and their costs
remain in `runs/diagnostics/dense_barrier_clean_stage4_pareto_screen_a20_phaseA34_phaseB220.npz`.

The barrier-threshold cost advantage at `0.017 kcal/mol` is `4.0x`; the
greater-than-10x claim is the exactly charged, matched-`I_perp` first-passage
comparison (`17.37x`).

## Fail-closed checks

All 19 checks pass, including experiment counts `20/20/10/10`, shared initial
path, complete ten-alpha grid, fixed alpha in both phases, reference-free
screen, exact counter consistency, alpha 20 as the unique target-reaching
member, exact stored complete macrostate, zero accepted-trace rises in both
metrics, superiority over ODE-NEB in both terminal metrics, at least 10x
charged `I_perp` speedup, explicit raw-shoulder disclosure, and all four final
figure stems present in PDF/SVG/PNG.
