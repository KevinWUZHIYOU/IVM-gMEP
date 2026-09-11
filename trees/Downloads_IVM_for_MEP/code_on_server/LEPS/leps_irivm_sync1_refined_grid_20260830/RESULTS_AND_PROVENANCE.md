# LEPS IR-IVM refined grid: results and provenance

## Scientifically selected parameters

The second-stage Cartesian grid was frozen before its first optimization run:

\[
\alpha\in\{20,50,100,200,500\},\qquad
r\in\{10^{-3},3\times10^{-3},10^{-2},3\times10^{-2},10^{-1}\},
\]

\[
m_{\max}\in\{3,4,5,6,8,10\}.
\]

All 150 registered combinations completed with zero run failures and consumed
99,720 Number of Forces in total.  The independent audit recomputed every
stored energy and gradient, both article metrics, and every direct/recovery
NoF subtotal.

The fixed selection rule chooses

\[
\boxed{\alpha=100,\qquad r=3\times10^{-3},\qquad m_{\max}=5.}
\]

The selected raw result is `runs/3c62d93a051be116.npz`.  It terminates at
1,558 NoF; its last accepted path is at 1,528 NoF.  The ledger closes as

\[
20\ \text{initial}+428\ \text{main}+555\ \text{direct reparameterization}
+555\ \text{single recovery}=1558.
\]

There are 28 completed outer iterations, 28 full direct reparameterizations,
and 28 single synchronized recovery attempts.  All 28 transactions roll back;
their force calls remain charged.

## Full-course result and interpretation

No combination satisfies the earlier 21/21 strict full-course acceptance
standard.  Fifty-seven of the 150 refined combinations pass 19/21 checks and
93 pass none.  The selected result misses only the densified barrier criterion
at NoF 150 and 278:

| NoF | available candidate state | candidate `I_perp` | article `I_perp` | candidate densified barrier error (eV) | article error (eV) |
|---:|---:|---:|---:|---:|---:|
| 150 | 135 | 14.7372733 | 16.3261955 | 3.048772069 | 3.035197033 |
| 278 | 218 | 14.4386993 | 22.1092676 | 2.700280300 | 2.535880054 |

Both metrics are strictly better than article IVM at NoF 66 and at every
registered check from NoF 468 onward.  Therefore this is the scientifically
ranked single-recovery LEPS IR-IVM, but it is not described as satisfying the
older full-course strict claim.

The parameter grid shows a broad stable region rather than an isolated tuned
number:

- `alpha=20` fails throughout, while `alpha=50, 100, 200`, and much of 500
  reach 19/21;
- `r` from `1e-3` through `1e-2` is effective, whereas `3e-2` and `1e-1`
  stop too early;
- `m_max=3` stops too early; values from 4 through 10 form a performance
  plateau;
- at `alpha=100, r=3e-3`, `m_max=5, 6, 8, 10` have identical metric scores
  and terminal NoF, so the registered tie-break selects the smallest,
  `m_max=5`.

## Selection across both registered stages

The initial 48-member grid and the refined 150-member grid contain 12
overlapping parameter combinations.  Their accepted path histories and NoF
histories are exactly equal.  After deduplication there are 186 unique tested
combinations, of which 63 pass 19/21 and zero pass 21/21.  Applying the same
fixed ranking to all 186 combinations selects the same refined-grid result:
`alpha=100, r=3e-3, m_max=5`.

Compared with the initial single-recovery choice `alpha=100, r=1e-3,
m_max=5`, the selected result retains the same 19/21 worst checkpoint ratio
but improves the mean joint ratio from 0.18667565 to 0.18588458 and reduces
terminal NoF from 1,587 to 1,558.

## Figures

The parameter-selection figure is
`output/pdf/leps_irivm_refined_grid_selection.pdf`.  The all-methods figure
using the selected trajectory is now the formal
`Figures/LEPS/leps_all_methods_metrics.pdf`, SHA-256
`9faaa4e643cd14329bdb65825f954886bb0232db04b0a425b7a7ae14a9051a25`.
The byte-identical source copy is
`code_on_server/LEPS/figures/leps_all_methods_metrics.pdf`.

By explicit user authorization, the selected IR-IVM curve permanently
replaces the former article IVM curve and is labelled simply `IVM`.  The old
article IVM is not plotted; the five other methods are unchanged.  The
19/21 selection status and the NoF 150 and 278 densified-barrier misses remain
recorded rather than being reclassified as a strict pass.  The complete
figure-level source registry is `FIGURE_PROVENANCE.md`.

## Traceable artifacts

- preregistration: `registrations/CAMPAIGN_PREREGISTRATION.json`
- protocol: `PROTOCOL.md`
- frozen-grid implementation wrapper: `src/leps_irivm_sync1_refined.py`
- serial runner: `src/run_grid.py`
- independent method and metric audit: `src/audit_and_select.py`
- two-stage deduplicated selection: `src/select_across_stages.py`
- campaign status: `reports/campaign_status.json`
- refined-grid selection: `reports/acceptance/selection_manifest.json`
- two-stage selection: `reports/acceptance/combined_selection_manifest.json`
- all 150 refined summaries: `reports/acceptance/candidate_summary.csv`
- all 21 same-NoF comparisons: `reports/acceptance/all_candidates_full_course.csv`
- NoF audit: `reports/acceptance/number_of_forces_audit.csv`
- selected raw result: `runs/3c62d93a051be116.npz`
- grid-figure source: `src/plot_refined_grid.py`
- grid-figure provenance: `reports/figure/leps_irivm_refined_grid_selection_provenance.json`
- all-methods source: `src/remake_leps_all_methods_selected.py`
- all-methods data: `reports/figure/leps_all_methods_metrics_source.csv`
- all-methods provenance: `reports/figure/leps_all_methods_metrics_provenance.json`
- formal figure registry: `FIGURE_PROVENANCE.md`

Selected SHA256 values:

| artifact | SHA256 |
|---|---|
| preregistration | `947cd467cf73ce18c230e951099f7406c6cd29e1c186dc2f17acc7e5439ce2f0` |
| refined implementation | `721fb2b77d6711dc032f0578ec12651064711c2adbe8afcc839228120148e971` |
| independent audit | `683e02261e4731537b41343efbc0443f2417f4ddff883deeba08bd368a9ac436` |
| refined selection manifest | `4ca04977a7331a93290a8f1b272b58213ef3f8c0bb502fa64d3c2611f44a1094` |
| two-stage selection manifest | `51e1979aa580afb1381cbe71573fa36f7db8cd04e208ed3097654b176ad53643` |
| selected raw result | `c7a09d0ae37cea45078dff2590e244390420c9b80931de1e4767e8b62f1bb02d` |
| grid-selection PDF | `effe8b59d73623f91875e20fa971252d43ff4ad2848199d42a87c59d99aaf0f0` |
| all-methods plotting source | `191cbb354c54aa5a1528d29454a01b877eed949dc8f740a6d982b9af486311db` |
| all-methods plotted-point CSV | `b0929cead1ae8c2e0d361a58ed66178ed69d13e8daad98eee0e270749643d78b` |
| all-methods reconstruction manifest | `e5e1558b35e1993acf3dbac136779f8560196b82c663be47d2ff42448ff5ab57` |
| formal all-methods PDF | `9faaa4e643cd14329bdb65825f954886bb0232db04b0a425b7a7ae14a9051a25` |
