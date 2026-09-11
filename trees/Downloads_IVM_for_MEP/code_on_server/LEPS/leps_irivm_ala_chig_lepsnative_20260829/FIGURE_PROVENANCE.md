# Provenance: `Figures/LEPS/leps_all_methods_metrics.pdf`

## Registered figure

The paper working copy and the LEPS source copy are byte-identical:

| Role | Path | SHA-256 |
|---|---|---|
| Paper figure | `Figures/LEPS/leps_all_methods_metrics.pdf` | `f558d0b0aeb66115bc3b36a015aad2cc2bfce748917b7e309abc19f6589b0376` |
| LEPS source figure | `code_on_server/LEPS/figures/leps_all_methods_metrics.pdf` | `f558d0b0aeb66115bc3b36a015aad2cc2bfce748917b7e309abc19f6589b0376` |

The one-page PDF was rendered with Poppler at 180 dpi after replacement and
visually checked for clipped labels, overlaps, missing curves, illegible
markers, and panel alignment.  No rendering defect was found.

The IR-IVM result contains 28 completed outer iterations in addition to the
common initial state.  All 29 states are present in the source CSV and all 29
are displayed with a blue circular marker (`marker_every = 1`).  Other
methods retain sparse markers for readability.  Thus no IR-IVM outer
iteration is hidden by marker subsampling.

## IR-IVM curve and strict acceptance

The plotted IR-IVM result is

`leps_irivm_ala_chig_lepsnative_20260829/runs/756bf4584a55e959.npz`

with SHA-256
`67cee4baa8c5ab05e6fe32c5a49dc0f2c3296d645ddfcbaf2765325dfa5b8047`.
Its preregistered controls are

\[
\alpha=100,\qquad r=10^{-2},\qquad m_{\max}=4.
\]

This result was not selected from an unregistered refinement.  It is the
winner of the complete 48-combination Cartesian grid

- `alpha = {10, 100, 1000}`;
- `r = {0.001, 0.01, 0.1, 0.3}`;
- `m_max = {4, 5, 6, 8}`.

All 48 combinations completed with zero execution failures and 36,215 total
joint energy-force evaluations across the campaign.  Fourteen combinations
passed the strict full-course gate.  The winner passed 21/21 pre-fixed
same-NoF checks for both metrics, with worst joint ratio
`0.9999990122401393` and mean joint ratio `0.13061944009688475`.  Its last
accepted path and terminal ledger both occur at NoF 2089.

The common initial path at NoF 20 is exactly equal to the article IVM path and
is excluded from the strict claim.  The formal checks are the 21 distinct
article-IVM accepted-path NoF values from 66 through 2572.  At every check,
the candidate path is selected by last observation carried forward and must
satisfy both strict inequalities:

\[
I_\perp^{\mathrm{IR-IVM}}<I_\perp^{\mathrm{article\ IVM}},
\qquad
|\Delta E_{500}^{\mathrm{IR-IVM}}-\Delta E_\mathrm{ref}|
<|\Delta E_{500}^{\mathrm{article\ IVM}}-\Delta E_\mathrm{ref}|.
\]

The independent acceptance manifest is
`reports/acceptance/selection_manifest.json`, SHA-256
`b9baef1104ab92d641b6d1c872f688679feb3a91686e2833da6ed99142b4edd8`.

## Algorithm represented by the curve

The method uses the current IR-IVM complete-normal proximal Armijo L-BFGS
subproblem and the Henkelman-Jonsson improved tangent everywhere.  It does
not use a central-difference tangent.  The transferred QM10 per-inner-step
cap is replaced by the LEPS article's deterministic total atomic trust
radius, `min(2 * median segment RMS length, 1)`.  This is not a searched
parameter.

Every five completed main stages, the method uses the Ala/Chignolin
identical-spring reparameterization and synchronized bicriteria recovery:
fixed recovery KKT fraction 0.8, one accepted recovery step per synchronized
trace, a 320-call per-image cap including the direct proposal, and atomic
commit or charged rollback.  Recovery does not read the tuned main `r`, and
there is no QM10 recovery routing or stage-dependent parameter schedule.

Roundoff-safe Armijo comparison and the Ala/Chignolin `I_perp`-monotone
complete-band commit rule are part of the frozen method.  The optimizer does
not read the article IVM trajectory, the reference saddle barrier, the
densified barrier, or the acceptance result.

The exact method and acceptance protocol are recorded in:

- `PROTOCOL.md`;
- `registrations/CAMPAIGN_PREREGISTRATION.json`, SHA-256
  `c02f3c5de96c44b6aeff5d824833f590e382fcd7c1ad18a55d52f09416e03a99`;
- `src/leps_irivm_lepsnative.py`, whose preregistered SHA-256 is
  `8e58791bc9c702de630dec8f034eab8c8b327f51150e7d8f7695e9dd9f6fdc1e`;
- `src/run_grid.py` for the serial, host-locked formal run.

## Metrics, Number of Forces, and other plotted methods

`I_perp` is independently recomputed with the article path-length quadrature
and improved tangent.  The plotted energy barrier is recomputed on 500 total
coordinate-wise linearly interpolated points.  The independent reference
barrier is `1.3430860763441048 eV`.

Number of Forces counts every uncached joint energy-force evaluation of the
20 internal images.  Direct reparameterization proposals, recovery trials,
and rolled-back transactions remain charged.  Rejected proposals are not
published as accepted algorithm states.  For every formal result, the sum of
the per-image ledgers equals the terminal NoF.

The audit artifacts are:

| Artifact | Purpose | SHA-256 |
|---|---|---|
| `reports/acceptance/candidate_summary.csv` | one row per grid combination | `dcf259cc95f8528d04425b7bdeca99de545281df47221c491cc5accaff7774fe` |
| `reports/acceptance/all_candidates_full_course.csv` | both metrics at every formal check | `6382c5cf62136dce34c0f450123b02267c81c3e21af795dec6677c92e9a5bc9f` |
| `reports/acceptance/number_of_forces_audit.csv` | per-run NoF closure and transaction subtotals | `743522b706f052cc631ed4f325ad3552dc7bc22e6355ed9a4214e3762f9706d3` |
| `src/audit_and_select.py` | independent method and metric audit | `318de23fbbca559477fc2d8c66043654aba20179bbf2930ab285177fa60108d9` |

The article IVM and the five other article-method curves are not regenerated
optimizer runs.  Their exact NPZ paths and SHA-256 values are enumerated in
`reports/figure/leps_all_methods_metrics_provenance.json`.

## Figure construction

The complete machine-readable plotted data are in
`reports/figure/leps_all_methods_metrics_source.csv`, SHA-256
`ea70c51120f3bf5e90cf0b7de696194740780da3654a6515d525724a42b6e471`.
It contains the plotted NoF, independently recomputed `I_perp`, densified
barrier, source-result path, and source-result SHA-256 for every point of all
seven curves.

The entry-point plotting code is `src/remake_leps_all_methods.py`, SHA-256
`20c49e8147072325b06404dc69d504ae73ce95f32948d0adce0175602086c696`.
It reuses the frozen article figure geometry and metric loaders from
`ir_ivm_qm10_v2p20_microrefine_20260829/src/remake_leps_all_methods.py`, whose
SHA-256 is recorded in the JSON figure provenance.  The exact reconstruction
manifest is `reports/figure/leps_all_methods_metrics_provenance.json`, SHA-256
`da04424347854beb3f53252c6d9e532fda524e2ce98df7f7f4dadafff03113da`.
