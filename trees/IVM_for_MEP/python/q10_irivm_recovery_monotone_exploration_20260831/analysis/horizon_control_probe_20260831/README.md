# Recovery-only horizon-first control-probe experiment

Date: 2026-08-31

Status: exploratory success.  This experiment does **not** modify the sealed
`q10_irivm_unified_cycle_frozen_reproduction_v2_20260831` campaign.

## Question and result

The exact recovery publication gate removed 30 of the 31 recorded increases
in public `I_perp`.  The remaining material increase was the N=20 recovery
trial at NoF 80:

- public anchor: `I_perp = 14975.056988037288`;
- ordinary local proposal `f=1`: `I_perp = 25360.379614046906`;
- its existing trust test accepted it provisionally and requested an
  expansion;
- the existing expansion `f=3`: `I_perp = 10528.695713273204`.

The local proposal therefore caused the increase even though the immediately
following, already-defined recovery expansion was strictly better.  Evaluating
the expansion first and retaining the local evaluation as a control probe
removes this increase without changing the state reached after both paid
bands.

The resulting single optimizer source gives:

| path resolution | controls `(alpha, r, m_max)` | public `I_perp` increases | strict `I_perp` checkpoints | strict densified barrier checkpoints | joint strict checkpoints | terminal NoF | NoF ledger |
|---|---:|---:|---:|---:|---:|---:|---:|
| N=10 | `(3e-3, 0.1, 7)` | 0 | 79/79 | 79/79 | 79/79 | 599 | 599 |
| N=20 | `(1e-3, 0.3, 8)` | 0 | 98/98 | 98/98 | 98/98 | 2363 | 2363 |

Both runs stopped at the fail-closed repeated-zero-NoF safety limit after the
last improving paid state.  Article checkpoints beyond the terminal paid state
were scored by the same registered last-observation-carried-forward rule.

## Unified online recovery rule

Let

`R(d) = -g^T d - ||d||^2 / (2 alpha)`

be the already-used regularized prediction, and let

`h = ceil(r m_max)`.

At a recovery anchor, the algorithm performs the following operations without
using an offline metric, an Article path, a path-resolution branch, or a NoF
stage branch.

1. Construct the ordinary `f=1` proposal and, when `h > 1`, the `f=h`
   proposal.  Both use the same warm recovery direction and the same formal
   improved-tangent nearest-neighbour spring reparameterization.
2. The ordinary choice is the proposal with the larger unscaled `R(d)`.
   A horizon-first trial is enabled only when normalizing each displacement by
   its own registered fraction reverses that choice, the horizon's unscaled
   `R(d)` is positive, and the reversal exceeds the existing machine-precision
   prediction floor.  No new fitted threshold is introduced.
3. Evaluate the horizon first.  Publish it only when its independently
   evaluated online `I_perp` is no larger than the current public envelope.
4. Evaluate the already-constructed local proposal from the frozen pre-horizon
   anchor as a control probe.  Do not publish it and do not write its secant.
5. Apply the existing trust test to the probe.  If it is trusted and
   non-monotone to the public anchor, the horizon has semantically replaced the
   existing `trusted local + expansion` pair and counts as two recovery
   commits.  Otherwise it counts as one.  Thus the probe controls the later
   recovery-cap timing rather than changing the accepted path.

For N=10, `ceil(0.1 * 7) = 1`, so the same source naturally reduces to the
ordinary single-proposal rule.  There is no `if N == 10` or `if N == 20`
algorithm branch.

## Why the 20 NoF control probe is used

The local probe is not an unaccounted or discarded trial.

- Its newly evaluated energies and forces supply the existing actual/predicted
  agreement test.
- That test decides whether the horizon consumes one recovery commit or
  replaces the two-commit local-plus-expansion transaction.  This decision
  changes when the recovery cap is reached and therefore changes subsequent
  optimizer control flow.
- Its secants are intentionally excluded.  Only the accepted anchor-to-horizon
  secants enter the physical L-BFGS memories, exactly as in the original
  completed expansion transaction.
- The 20 calls are charged to `recovery_band_trial`; the per-image sum and the
  category ledger both close exactly.

At the only N=20 horizon-first event, the paid horizon is public at NoF 80.
The local control probe completes at NoF 100 with:

- `candidate_i = 25360.379614046906`;
- `trusted = true`;
- `monotone_to_public = false`;
- `new_force_calls = 20`;
- `recovery_commit_increment = 2`.

The public state remains the accepted horizon at NoF 100.  At the Article
checkpoints:

| NoF | candidate `I_perp` | Article `I_perp` | candidate barrier error | Article barrier error |
|---:|---:|---:|---:|---:|
| 80 | 10528.695713273204 | 25473.3830778931 | 18.7639119416518 | 37.23040103473461 |
| 100 | 10528.695713273204 | 18807.020024169196 | 18.7639119416518 | 31.41144534625171 |

## State-equivalence audit after NoF 100

The N=20 run was rerun in one process against the exact-public-envelope
reference trajectory with the same controls.  From NoF 100 onward:

- public-state counts: 112 versus 112;
- public NoF arrays: exactly equal;
- path arrays: bitwise equal, maximum absolute difference `0.0`;
- stored energy arrays: bitwise equal, maximum absolute difference `0.0`;
- stored gradient arrays: bitwise equal, maximum absolute difference `0.0`;
- final per-image L-BFGS memories: 300 versus 300 pairs and bitwise equal;
- terminal NoF: 2363 versus 2363;
- complete NoF category breakdowns: equal.

The only intentional metadata difference is the phase label at NoF 100:
`horizon_local_control_probe_hold` records the reordered control transaction.
It does not change the numerical state.

Reference trajectory used for this equivalence check:

`runs/exact_envelope_n20_a0p001_r0p3_m8/trajectory.npz`

SHA-256:
`54828893a8a8d8de3edff367651595f2b59781f6bf9004b5d1392eb995d0c40a`

## Hard-condition audit

- Only recovery proposal ordering, recovery publication, and recovery commit
  bookkeeping are changed.  The main subproblem and post-main solve are not
  changed.
- Every reparameterization remains
  `(L[i+1] - L[i]) * tau_i`, with `tau_i` from the improved tangent, spring
  step `0.05`, at most 500 iterations, and tolerance `1e-5`.
- Recovery publication uses the exact online public `I_perp` envelope; there is
  no additive `1e-14`-scale slack at the numerical floor.
- The same source is used for N=10 and N=20.
- There is no condition on path resolution, current NoF, training stage,
  Article data, reference barrier, densified barrier, or checkpoint outcome.
- The only main controls remain `alpha`, `r`, and `m_max`; `h` is derived from
  the registered `r` and `m_max`.  The prediction floor is a pre-existing
  floating-point resolution guard, not a tuned control.
- The selected controls preserve `r < 0.7`, `m_max >= 4`, and integer-mantissa
  scientific-notation `alpha` values.

A static search of the optimizer found no comparisons against `n_internal` or
`ledger.total`, and no `REFERENCE_BARRIER`, Article, densified-Delta-E,
checkpoint, training, or baseline input in the algorithm.

## Artifacts and hashes

All paths below are relative to
`python/q10_irivm_recovery_monotone_exploration_20260831/`.

Optimizer source:

- `src/q10_horizon_control_probe_optimizer.py`
- SHA-256:
  `3fcffd7df4d5d92cacafa7ba835e9ca748c57bcd8d7c5f6baa6d1a6ba6147342`

Scoring entry point:

- `src/run_and_score.py`, variant `horizon_control_probe`

N=10 artifacts:

- `runs/horizon_control_probe_n10_a0p003_r0p1_m7_b3000_subagent/summary.json`
- summary SHA-256:
  `97bdf86d2facd3726aefb5045eb7bcbba90b98e9e858e42f174e37afd5c2460a`
- `runs/horizon_control_probe_n10_a0p003_r0p1_m7_b3000_subagent/trajectory.npz`
- trajectory SHA-256:
  `921163458325091d7ef260d41d74bc6241b0a853c7410835be81055b0ea7c200`

N=20 artifacts:

- `runs/horizon_control_probe_n20_a0p001_r0p3_m8_b3000_subagent/summary.json`
- summary SHA-256:
  `ae178584b439369f2e3a829e46afa61a518152f443438cb01f59aa944a99920d`
- `runs/horizon_control_probe_n20_a0p001_r0p3_m8_b3000_subagent/trajectory.npz`
- trajectory SHA-256:
  `9799c7108f7772db632670637e1b555bfcd6b45eb9cb1669d7183885fe92b705`

## Reproduction commands

```bash
env -u OMP_NUM_THREADS -u MKL_NUM_THREADS -u VECLIB_MAXIMUM_THREADS \
  -u NUMEXPR_NUM_THREADS OPENBLAS_NUM_THREADS=8 PYTHONHASHSEED=0 \
  /Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  -m python.q10_irivm_recovery_monotone_exploration_20260831.src.run_and_score \
  --variant horizon_control_probe --n 10 --alpha 3e-3 --r 0.1 --m 7 \
  --budget 3000 --label-suffix _reproduction

env -u OMP_NUM_THREADS -u MKL_NUM_THREADS -u VECLIB_MAXIMUM_THREADS \
  -u NUMEXPR_NUM_THREADS OPENBLAS_NUM_THREADS=8 PYTHONHASHSEED=0 \
  /Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  -m python.q10_irivm_recovery_monotone_exploration_20260831.src.run_and_score \
  --variant horizon_control_probe --n 20 --alpha 1e-3 --r 0.3 --m 8 \
  --budget 3000 --label-suffix _reproduction
```

## Zero-rise comparison figure

The focused N=20 figure overlays Article IVM, the exact-envelope predecessor
before proposal reordering, and the horizon-first control-probe result.  The
main axes show the complete trajectories; the insets isolate NoF 20--140 and
mark the NoF 80 replacement.  The energy panel uses the article metric
`|Delta E_linear500 - Delta E_ref|`, not the raw barrier.

- PDF: `output/pdf/q10_irivm_zero_rise_n20_comparison.pdf`
- PNG: `output/pdf/q10_irivm_zero_rise_n20_comparison.png`
- plot source: `analysis/horizon_control_probe_20260831/plot_zero_rise_comparison.py`
- plotted source table: `analysis/horizon_control_probe_20260831/q10_irivm_zero_rise_n20_source.csv`
- figure manifest: `analysis/horizon_control_probe_20260831/q10_irivm_zero_rise_n20_figure_manifest.json`

The final Poppler render passed visual inspection with no clipping,
overlapping labels, missing glyphs, or unreadable annotations.  The focused
figure uses N=20 because N=10 is bitwise unchanged by this recovery rule.
