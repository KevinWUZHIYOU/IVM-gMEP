# LEPS single synchronized recovery: result and provenance

## Result

This experiment does not remove reparameterization.  After every completed
outer iteration it performs the following atomic transaction:

1. construct the full improved-tangent spring reparameterization proposal;
2. evaluate that direct proposal and charge all new Number of Forces;
3. commit it immediately if the original path-convergence and equidistance
   guards pass;
4. otherwise form one synchronized complete-band normal recovery direction
   from the already evaluated direct gradients;
5. choose one common step using the regularized predicted decrease and then
   evaluate the resulting recovery band at most once;
6. apply the unchanged bicriteria merge and either commit the complete
   transaction or roll back to the pre-reparameterization state.

Thus recovery is always conditional on an explicit direct reparameterization.
Rejected direct and recovery proposals are still charged, but are not exposed
as accepted states in the performance curve.

The preregistered 48-parameter grid completed with zero run failures and a
cumulative 25,179 Number of Forces.  Independent recomputation found no
parameter combination that satisfies the full strict acceptance rule.  The
best result is

\[
\alpha=100,\qquad r=10^{-3},\qquad m_{\max}=5,
\]

with 19 of the 21 same-NoF checks strictly better than article IVM in both
metrics.  It misses only NoF 150 and NoF 278, and at both checks only the
densified barrier error is worse.  Both metrics are strictly better at every
registered check from NoF 468 onward.  The article comparison figure is
therefore not replaced.

## Number-of-Forces and transaction audit

The selected result contains 28 completed outer iterations, 28 direct
reparameterization trials, and 28 single synchronized recovery transactions.
All 28 transactions roll back.  Its terminal Number of Forces is 1,587 and
the independently closed ledger is

\[
20\ \text{(initial)}+449\ \text{(main stages)}
+559\ \text{(direct reparameterization)}
+559\ \text{(single recovery)}=1587.
\]

The single-recovery change substantially reduces the cost that caused the
large staircase in the earlier period-one experiment:

| method | controls | outer iterations | transactions | direct NoF | recovery NoF | mean NoF per transaction | checks passed |
|---|---|---:|---:|---:|---:|---:|---:|
| accepted recovery every 5 iterations | `alpha=100, r=1e-2, m_max=4` | 28 | 5 | 100 | 1,565 | 333.0 | 21/21 |
| repeated recovery every iteration | same controls | 10 | 10 | 200 | 2,196 | 239.6 | 1/21 |
| single recovery every iteration | same controls | 28 | 28 | 556 | 556 | 39.7 | 19/21 |
| selected single recovery every iteration | `alpha=100, r=1e-3, m_max=5` | 28 | 28 | 559 | 559 | 39.9 | 19/21 |

Each non-cached transaction is capped at 20 direct plus 20 recovery Number of
Forces.  A fully cached proposal may execute while adding zero NoF.  The audit
therefore checks the correct one-way relation: positive recovery NoF requires
a recorded synchronized trial, while a recorded synchronized trial need not
add NoF if all requested states are cached.

## Equal-NoF details for the selected result

| checkpoint NoF | available accepted state NoF | candidate `I_perp` | article `I_perp` | candidate densified barrier error (eV) | article error (eV) | result |
|---:|---:|---:|---:|---:|---:|---|
| 66 | 59 | 14.7653057 | 19.9946623 | 3.143191851 | 3.143194956 | both better |
| 150 | 140 | 14.7371907 | 16.3261955 | 3.048772069 | 3.035197033 | barrier fails |
| 278 | 225 | 14.4386069 | 22.1092676 | 2.700280300 | 2.535880054 | barrier fails |
| 468 | 407 | 0.1202684 | 7.5386670 | 0.002486687 | 0.558782458 | both better |
| 666 | 643 | 6.8165e-7 | 0.1490245 | 0.000994652 | 0.026363760 | both better |
| 1052 | 1040 | 9.0904e-16 | 0.00342675 | 0.000995354 | 0.018846588 | both better |

The comparison figure uses last-observation-carried-forward at the registered
article NoF checkpoints.  Direct or recovery proposals that fail the atomic
commit are never plotted as accepted paths.

## All-methods candidate preview

The existing LEPS all-methods figure geometry, typography, article methods,
metrics, and 20,000-NoF axes were reused with only its IR-IVM trajectory
changed to the selected single synchronized recovery result.  The resulting
candidate preview is
`output/pdf/leps_all_methods_metrics_single_sync_preview.pdf`.

This is deliberately not installed over `Figures/LEPS/leps_all_methods_metrics.pdf`:
the previewed trajectory passes 19/21 rather than 21/21 registered checks.
Its independent figure data and provenance are stored as
`reports/figure/leps_all_methods_metrics_source.csv` and
`reports/figure/leps_all_methods_metrics_provenance.json`.

## Traceable artifacts

- preregistration: `registrations/CAMPAIGN_PREREGISTRATION.json`
- protocol: `PROTOCOL.md`
- implementation: `src/leps_irivm_sync1.py`
- serial grid runner: `src/run_grid.py`
- preregistered independent audit: `src/audit_and_select.py`
- cache-aware audit correction: `src/audit_and_select_cacheaware.py`
- campaign status: `reports/campaign_status.json`
- independent selection result: `reports/acceptance/selection_manifest.json`
- all 48 parameter summaries: `reports/acceptance/candidate_summary.csv`
- all same-NoF comparisons: `reports/acceptance/all_candidates_full_course.csv`
- Number-of-Forces audit: `reports/acceptance/number_of_forces_audit.csv`
- selected raw result: `runs/356307bb5a618535.npz`
- comparison plot source: `src/plot_sync1_comparison.py`
- comparison PDF: `reports/figure/leps_single_synchronized_recovery_comparison.pdf`
- comparison PNG: `reports/figure/leps_single_synchronized_recovery_comparison.png`
- all-methods preview source: `src/remake_leps_all_methods_preview.py`
- all-methods preview PDF: `output/pdf/leps_all_methods_metrics_single_sync_preview.pdf`
- all-methods preview data: `reports/figure/leps_all_methods_metrics_source.csv`
- all-methods preview provenance: `reports/figure/leps_all_methods_metrics_provenance.json`

Selected SHA256 values:

| artifact | SHA256 |
|---|---|
| preregistration | `add172f7604d106546e7c869a3a6e01a37539ceb455f9985c65402b71f2ba093` |
| implementation | `8b1f848d6a2f03090b4851b6bd4a45d55a61fbee4025dfa50f5e59e2084ea49a` |
| preregistered audit | `e6b85ff03656b20ca0cf9b63f43c046a189869c4bb04d5e2065c474ad85cdbac` |
| cache-aware audit | `5ebe35e961ccb7e0d35f97a51bdf110dcb76341bb4a7091ea30edfe93939c32b` |
| selection manifest | `282ebaa9ab7c7c65c222c5ef66b4dceb41e165cb264d8ca1bcb11e07090596e3` |
| selected result | `3b15974315d318dce5c2cdc0cf0f7d9474cce3936c03a2fb581af5c47092d740` |
| comparison plot source | `ffb3a2e32575464ce83b27a2ee369abf92f2d35396af7a8c2743b62e4808aea1` |
| comparison PDF | `42d1a8b3caad27d6139b31059a786a61119ff87d907b0e3a40d7854981273576` |
| comparison PNG | `a158346c39c94ced95dca29800db4cc79cad568f92513a15ea3fe411bb6d6a19` |
| all-methods preview source | `d75d66af46fadd638bbf0a5ee120d0d4c8fbe5cdfb8c56cfd08fb729fa88c884` |
| all-methods preview PDF | `1a03ee2f97e14bc92b08206bc85a9ea40d98c0fe202d4fefa4a1fc56427bb0c6` |
| all-methods preview data | `b63126886b5ce8025d1285b6d6c94246abd25ed61586d961fb5dd72516bc0d12` |
| all-methods preview provenance | `2588806d513f55058ce56ae338a1662c719c5c5c3d98413a450b3e6193ea2fde` |

The preregistered audit source is preserved unchanged.  Its only post-run
problem was an invalid equivalence between a recorded synchronized trial and
positive new recovery NoF in a fully cached case.  The separately retained
cache-aware audit changes only that invariant; it does not change or rerun the
optimizer, raw results, metrics, checkpoints, or selection rule.
