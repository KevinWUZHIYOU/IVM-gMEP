# QM10 low-r IR-IVM strict-comparison provenance

## Figure and scientific claim

The figure `/Users/wuzhiyou/IVM_for_MEP/output/pdf/q10_irivm_low_r_strict_comparison.pdf`
compares the formally audited low-r IR-IVM winners with the frozen Article IVM
for `N_internal=10` and `N_internal=20`.  It plots the complete public-state
trajectories of the two article metrics against Number of Forces:

- Henkelman--Jonsson improved-tangent `I_perp`;
- `abs(Delta E_dense - 67.50188060819488)`, where `Delta E_dense` uses the
  coordinate-wise 500-point linearly densified path.

The shaded interval in each column is the registered latter-half comparison
range.  A dashed IR-IVM segment means that the last public state is carried
forward by LOCF after the optimizer has stopped; it does not imply additional
force evaluations.

This is disclosed post-hoc QM10 method development against a known opponent,
not an independent transfer or holdout result.  The supported claim is narrow:
the same unified IR-IVM algorithm, using only `r < 0.7`, has at least one
setting at each registered resolution that is strictly lower than Article IVM
in both metrics at every one of the 51 latter-half same-NoF checkpoints.

## Registered experiment and selected runs

Campaign:
`q10-irivm-low-r-coherence-secant-v2-20260830`.

The shared registered grid was

```text
alpha labels = {1e-3, 2e-3}
r            = {0.60, 0.65}
m_max        = {4}
N_internal   = {10, 20}
```

Thus all alpha labels have integer mantissas, all settings satisfy `r < 0.7`,
and the two resolutions use the same source and the same four parameter
triples.  Parameters do not change within a run.

| Resolution | Selected controls | Run ID | Terminal NoF | Strict result | Worst `I_perp` ratio | Worst barrier-error ratio |
|---|---|---|---:|---|---:|---:|
| `N_internal=10` | `alpha=2e-3`, `r=0.65`, `m_max=4` | `c5a578f9a8b87197` | 847 | 51/51 in both metrics | 0.0262908426 | 0.9996979797 |
| `N_internal=20` | `alpha=1e-3`, `r=0.65`, `m_max=4` | `c14cc0cf2a7f6171` | 2282 | 51/51 in both metrics | 0.1827990490 | 0.9907478520 |

The `N_internal=20`, `alpha=1e-3`, `r=0.60`, `m_max=4` run also passes 51/51
in both metrics and stops at 2276 NoF.  The registered ranking names `r=0.65`
the winner only because its mean joint ratio is smaller by about
`1.48e-13`; this is not a practically meaningful advantage over `r=0.60`.

The closest strict margins are:

- `N_internal=10`, checkpoint NoF 481: barrier error
  `0.03783392874135` versus Article `0.03784535880749`, an absolute margin of
  `1.1430066e-5` and a relative margin of about `0.0302020%`;
- `N_internal=20`, checkpoint NoF 2330: barrier error
  `0.001867747523590` versus Article `0.001885189576470`, an absolute margin
  of `1.7442053e-5` and a relative margin of about `0.9252148%`.

## Exact data sources

Selected IR-IVM results and receipts:

- `runs/n10/c5a578f9a8b87197.npz` and
  `reports/receipts/n10/c5a578f9a8b87197.receipt.json`;
- `runs/n20/c14cc0cf2a7f6171.npz` and
  `reports/receipts/n20/c14cc0cf2a7f6171.receipt.json`.

Frozen Article IVM results:

- `/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/muller_q10_10d/image12/data/frozen_runs/tuned_ivm_alpha2e3_selected.npz`
  for `N_internal=10`;
- `/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/muller_q10_10d/data/frozen_runs/tuned_ivm_alpha1e3_selected.npz`
  for `N_internal=20`.

Machine-readable selection and checkpoint values:

- `reports/result_audit/audit_manifest.json`;
- `reports/result_audit/candidate_summary.csv`;
- `reports/result_audit/checkpoint_comparison.csv`;
- `reports/result_audit/AUDIT_REPORT.md`.

Absolute paths and SHA-256 hashes for the PDF, plotting source, selected NPZ
and receipt files, Article data, registration, metric source, N20 gate, final
audit, and provenance document are recorded in
`reports/selected_figure_manifest.json`.

## Algorithm represented by IR-IVM

Both resolutions run one unified source with Henkelman--Jonsson improved
tangents everywhere, a complete-normal proximal Armijo L-BFGS main subproblem,
improved-tangent identical-spring direct reparameterization after every outer
stage, and a whole-band atomic recovery gate.  No central-difference tangent,
resolution-specific algorithm branch, NoF-specific branch, or training-phase
parameter schedule is allowed.

Only accepted-recovery memory transport differs from the frozen
chord-coherent predecessor.  A complete accepted recovery certifies physical
direct-to-projected secants using
`s^T y > 1e-6 ||s|| ||y||`.  A strong chord-coherent recovery publishes only
the certified pair with greatest normalized curvature; conservative and
ordinary stationarity recoveries publish all certified pairs.  This adds no
tunable control and uses no force evaluation.

Winner NoF accounting closes as follows:

- `N_internal=10`: `847 = 10` initial + `489` main + `130` direct
  reparameterization misses + `218` projected-recovery misses;
- `N_internal=20`: `2282 = 20` initial + `1452` main + `300` direct
  reparameterization misses + `510` projected-recovery misses.

Raw recovery construction, spring/polyline seeding, chord projection algebra,
secant certification, and ranking cost zero NoF.  Every uncached internal-image
energy-gradient evaluation, including rejected and rolled-back trials, costs
one NoF.

## Audit and staged-resolution gate

All eight registered results are integrity-valid; there are zero audit
failures.  The N20 runs were completed and independently audited first.  The
four N10 NPZ and receipt files carry the SHA-256 of that exact N20 gate
manifest, while all N20 files carry `not_applicable`.  The final audit then
reopened all eight results, their receipts, all registered dependencies, and
both frozen Article runs.

The independent audit reconstructs the chronological exact-coordinate cache
miss trace, per-image and terminal NoF ledgers, outer-start/post-main/committed
state chain, direct spring reparameterization, recovery seed and
improved-tangent chord-Newton projection, raw-step and fixed-half retry
algebra, recovery acceptance, secant publication, termination reason, both
metrics, LOCF states, and every strict checkpoint comparison.

## V1 audit correction and v2 status

The preserved v1 campaign ran N20 first but did not release its N10 gate.  Its
independent audit reconstructed a late recovery direction from `raw-direct`
after coordinate addition.  At outer 67 the true logged `g^T d` was about
`-1.88e-25`, but subtracting two order-one coordinates whose difference was
only about one ULP changed the reconstructed sign.  The audit therefore
reported a false integrity failure even though the v1 N20 path reproduced
51/51 strict wins in both metrics.

V2 was preregistered after that failure was observed and is explicitly not a
holdout.  Its numerical optimizer and control flow are unchanged.  The only
extra executable operation is a write-only, zero-NoF copy of the recovery
direction before coordinate addition.  The independent audit uses that field
to recompute `g^T d`, `||d||^2`, the clipped step, prediction, and raw path.
AST tests prove that the direction log is not read by the optimizer and that
no numerical or control-flow change was introduced.

## Reproduction chain

1. Read the frozen method, grid, evidence disclosure, NoF contract, and strict
   rule in `PROTOCOL.md`.
2. Verify every pre-run dependency hash and the integer-mantissa alpha contract
   in `registrations/CAMPAIGN_PREREGISTRATION.json`.
3. Reproduce N20 with `src/run_grid.py --jobs 1 --n-internal 20` under the
   registered single-thread environment.
4. Create the staged gate with
   `src/audit_results.py --n-internal 20 --output-dir reports/n20_gate_audit`.
5. Reproduce N10 with `src/run_grid.py --jobs 1 --n-internal 10`; the runner
   refreshes and authenticates the N20 gate before optimization.
6. Audit all eight runs with
   `src/audit_results.py --output-dir reports/result_audit`.
7. Generate the figure with `src/plot_selected_comparison.py`.  The script
   reopens the final audit winner, result, receipt, gate token, Article data,
   and registered metric source; recomputes the plotted metrics; and refuses
   to draw unless both selected runs reproduce 51/51 strict joint wins.
8. Verify exact paths and hashes in `reports/selected_figure_manifest.json`.

The delivered PDF was rendered back to PNG with Poppler at 180 dpi and visually
checked for clipping, overlap, legibility, and correct panel labels.
