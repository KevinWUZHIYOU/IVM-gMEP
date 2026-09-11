# Alanine 66D fixed-alpha 2x3 cost-to-precision figure provenance

## Registration status

**Complete, read-only validated, and visually verified.**  The registered PDF
compares three frozen Ala IVM settings over the same decade grid
\(\alpha=10^{-1},10^0,\ldots,10^6\) and the same 5000-force optimization
budget.  The PDF, exact plotted-record table, and machine-readable manifest
were generated from the formal inputs below.  The earlier 2x2 figure and all
historical alpha figures remain unchanged.

Stable stem:

`alanine66d_alpha_cost_to_precision_2x3`

| Artifact | Location | SHA-256 |
|---|---|---|
| PDF | `output/pdf/alanine66d_alpha_cost_to_precision_2x3.pdf` | `bd988a1aeaeb273853a06cb5f006526429b75d96405427b5ab82abe67f0e078d` |
| Plotted-record table | `output/pdf/alanine66d_alpha_cost_to_precision_2x3_source.csv` | `93e1e3910ef3a0c224833171d420714b9edd61d144e92bdafd7f31e7e52a6814` |
| Machine-readable manifest | `output/pdf/alanine66d_alpha_cost_to_precision_2x3_manifest.json` | `8384d01b902a53b495ce5ca41130f73916c19731d112dcae4b782589b87b1366` |
| 2x3 plotter | `paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_alpha_cost_to_precision_2x3.py` | `4768df3b0a407a7b9a0d4bc96400cee0e5b831c8f3e66ce8cdf5e02e0569e125` |
| Imported 2x2 companion | `paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_alpha_cost_to_precision_2x2.py` | `6d379347492872d4ef68f92b5b4fbcc46f9511add285adb73542fa42a8885b28` |

The manifest additionally freezes every optimizer-result hash, every
referenced dense-metric artifact hash, and the registered implementation and
input hashes used by the three columns.

## Scientific scope and controlled comparisons

| Column | Inner-solve controls | Reparameterization | Saved-path rule |
|---|---|---|---|
| left | \(r=0.1\), \(m_{\max}=5000\), safety cap certified nonbinding | disabled | every formally complete saved path; mesh quality recorded but not used as a gate |
| middle | \(r=0.1\), \(m_{\max}=5000\), safety cap certified nonbinding | enabled after every outer solve | initial or reparameterization-committed, formally complete, common-mesh-eligible paths |
| right | production setting \(r=0.7\), \(m_{\max}=5\) | enabled after every outer solve | initial or reparameterization-committed, formally complete, common-mesh-eligible paths |

Thus the left--middle comparison holds \(r=0.1\) and the nonbinding
implementation safety cap fixed and changes reparameterization.  The
middle--right comparison retains reparameterization and changes the inner
stopping rule.  All columns use the same common IDPP initial path, discrete
Ala IVM family, alpha grid, and force budget.

There is one essential qualification.  The no-reparameterization metric
campaign preregistered **all 442 formally complete saved states**, but only
**11/442** satisfy the common mesh criterion
\(\operatorname{CV}\le0.0065\) and
\(\max\Delta s_i/\min\Delta s_i\le1.03\).  Its mesh gate is explicitly
`not_applied`; 431 states are mesh-ineligible.  Consequently, the left column
is a **descriptive display of the actual no-reparameterization trajectory**.
In particular, a small raw \(I_\perp\) there must not be presented as a strict
causal improvement measured on the same mesh as the reparameterized columns.
The independently densified barrier and the retained mesh diagnostics are
needed to expose that distinction.

## Frozen formal inputs

### Left: \(r=0.1\), without reparameterization

The eight optimizer calculations were completed by the formal
`exact_no_reparam` family before the descriptive dense-metric campaign was
registered.

| Optimizer input | Location | SHA-256 |
|---|---|---|
| preregistration | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_grid_campaign_successor_v2_1787604432554391000_preregistered.json` | `d3ebf3239931dfde104342b791d8ed096b4ddef597c1bee0de4cb32b4b289aa3` |
| completed exact/no-reparameterization aggregate | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_grid_campaign_successor_v2_1787604432554391000_exact_no_reparam_aggregate.json` | `a2307bd55d7b165d07e3d83c61c6c5b41f5449b7db72d66f540ffaf74cf0556e` |
| local worker report | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_grid_campaign_successor_v2_1787604432554391000_exact_no_reparam_local_worker.json` | `8d1f75527ac04402b05fe08220a24f8ed24043853888d9cafa9d58a091f44c48` |
| remote worker report | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_grid_campaign_successor_v2_1787604432554391000_exact_no_reparam_remote_worker.json` | `640947cdb4c3981e3b5950235eba2d2f9ec8497336b1e7d5a662e7e06580bccb` |

| Dense-metric input | Location | SHA-256 |
|---|---|---|
| preregistration | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_exact_noreparam_alpha_r01_c5000_metric_campaign_v1_1788944303113946000_preregistered.json` | `cd44dbf802e703abb3509722932c0d6aa13c57a731db6b2f9025829205db746b` |
| completed aggregate | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_exact_noreparam_alpha_r01_c5000_metric_campaign_v1_1788944303113946000_aggregate.json` | `5a436dea9d6e3f47c5971c27a292bba8f65b2230cc2cb737e080e6a2e14044cc` |
| local run report | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_exact_noreparam_alpha_r01_c5000_metric_campaign_v1_1788944303113946000_local_run_report.json` | `ab94416ab7c2c99f6d9f7f42bb308b3756e7b6f5faf3a82ba4f1c100c5519d77` |
| metric wrapper | `python/alanine_ivm_clean_four_level_20260801/src/run_stage4_ala_discrete_exact_noreparam_alpha_r01_c5000_metric_campaign.py` | `59987a7197863dd9f25b72ca07fb978e34cabf293b4b35257106dee1b7f39c53` |

The metric aggregate records 442 saved-state references and 435 unique paths.
Four path metrics were reused by exact SHA-256 from metric-v2 and 431 were
evaluated locally.  All 442 selected level-160 barriers passed the registered
level-160 versus level-80 convergence test.  The new diagnostics used
1,448,591 raw joint energy evaluations and the reused diagnostics account for
13,444 historical evaluations; all are offline diagnostics and add zero to
the plotted optimization cost.

### Middle: \(r=0.1\), with reparameterization

| Optimizer input | Location | SHA-256 |
|---|---|---|
| preregistration | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_campaign_v1_1788923892307558000_preregistered.json` | `da1ded53f6dc11ac71ebe5bae64ba452f19c95fdc60467be220ff630912b6a05` |
| completed aggregate | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_campaign_v1_1788923892307558000_aggregate.json` | `50914e1cdfdafc59287921b4eef6081a5274d40366a4da926a590c06a6c9828a` |
| local report | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_campaign_v1_1788923892307558000_local_report.json` | `78ef8b82b7b7a8dbc2b6f4f62d8728fb216c1494fce46a95bbddce5241592362` |

| Dense-metric input | Location | SHA-256 |
|---|---|---|
| preregistration | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_alpha_r01_c5000_metric_campaign_v1_1788924371071068000_preregistered.json` | `540b6a985d09219cb112093003ce7f63de709cd3cb071a9f9e78645a2dd201a4` |
| completed aggregate | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_alpha_r01_c5000_metric_campaign_v1_1788924371071068000_aggregate.json` | `0f7a5c9220dace0468e0f1edd56a4693de71526d9b7f1682ad5ddc3f384a2295` |
| local run report | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_alpha_r01_c5000_metric_campaign_v1_1788924371071068000_local_run_report.json` | `f5f7fdf368faec673576ac2cb36f83f84d6a7059fa814d1f1f91897e8b2e0d22` |

The optimizer aggregate records eight unique completed runs and certifies
that the \(m_{\max}=5000\) safety cap was nonbinding in every run.  The metric
aggregate contains 121 formal-complete, common-mesh-eligible state references
covering 114 unique paths.  All dense metrics were reused by exact path hash,
all converged, and no diagnostic evaluation was charged to optimization
Number of Forces.

### Right: production \(r=0.7,m_{\max}=5\), with reparameterization

| Dense-metric input | Location | SHA-256 |
|---|---|---|
| preregistration | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_grid_metric_campaign_v2_1787608620070816000_preregistered.json` | `5fcfad3228e72c267ea1d768dd6f677528ae1fc8e8c6f180ea0d3498503b98a3` |
| completed aggregate | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_grid_metric_campaign_v2_1787608620070816000_aggregate.json` | `0e53a4e00a8a30f8e60c9fd278426f9b55fecada08840dee6f514a4e93390e63` |
| local host report | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_grid_metric_campaign_v2_1787608620070816000_local_host_report.json` | `59c55a351f9babe66ed4a91bf16e94c4291e0883a5fe6f2213982b915e0b9f83` |
| remote host report | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_grid_metric_campaign_v2_1787608620070816000_remote_host_report.json` | `0c18528c97224ea458d717f34d9d4e113cf1ceff7b18cf7c6b413d868f21c248` |

The plotter selects exactly the eight production-setting alpha runs from this
larger frozen campaign: 258 formal-complete, common-mesh-eligible state
references.  Their optimizer-result and task-artifact identities are frozen
in the final figure manifest.  The metric-v2 aggregate certifies all path
tasks converged and that its offline diagnostics add zero to optimization
Number of Forces.

## Saved-path, metric, and cost rules

The upper row is raw \(I_\perp\).  The lower row is the actual independently
densified energy barrier \(\Delta E\), not its error from the reference.

For each alpha, records at the same observed Number of Forces are first
resolved by retaining the lowest actually observed metric value.  The curve
then connects only strict improvements in increasing cost.  There is no
interpolation between observations, no extrapolation beyond the last record,
and no carry-forward to the 5000-force budget.  The source CSV contains 1,415
plotted vertices: left 442/306, middle 121/98, and right 258/190 for raw
\(I_\perp\)/densified \(\Delta E\), respectively.

The ordinate is the optimizer field `total_internal_joint_evaluations`, i.e.
force evaluations at the moving images; fixed endpoints are excluded.  The
independent densified-path calculations are tracked separately and are not
added to Number of Forces.

Every barrier uses piecewise-linear interpolation in Cartesian coordinates
on nested grids with 20, 40, 80, and 160 subdivisions per original segment.
All four levels are evaluated, level 160 is always selected, and
\(|\Delta E_{160}-\Delta E_{80}|\le10^{-3}\)
\(\mathrm{kcal\,mol^{-1}}\) is required.  The definition is

\[
\Delta E=\bigl(\max E_{\mathrm{dense}}-E_{\mathrm{dense},0}\bigr)
\times 23.0605\ \mathrm{kcal\,mol^{-1}}.
\]

The reference line is
\(\Delta E_{\mathrm{ref}}=2.4560825131253083\,
\mathrm{kcal\,mol^{-1}}\).  The plotter fails if a no-reparameterization
path undershoots this reference by more than \(10^{-12}\)
\(\mathrm{kcal\,mol^{-1}}\); the registered input passed that gate.

## Display and validation

All six panels share the logarithmic Number-of-Forces range 18--5500.  The
reversed metric windows are \(I_\perp:10^3\to10^{-4}\) and
\(\Delta E:4.0\to2.44\,\mathrm{kcal\,mol^{-1}}\).  The eight alpha values
use one marker/color legend and the discrete `jet` color map from blue to red
as alpha increases.

Read-only validation and figure generation use:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/ala_alpha_ctp_2x3_mpl \
XDG_CACHE_HOME=/private/tmp/ala_alpha_ctp_2x3_cache \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_alpha_cost_to_precision_2x3.py \
  --noreparam-metric-preregistration \
  python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_exact_noreparam_alpha_r01_c5000_metric_campaign_v1_1788944303113946000_preregistered.json \
  --noreparam-metric-aggregate \
  python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_exact_noreparam_alpha_r01_c5000_metric_campaign_v1_1788944303113946000_aggregate.json \
  --reparam-r01-metric-preregistration \
  python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_alpha_r01_c5000_metric_campaign_v1_1788924371071068000_preregistered.json \
  --reparam-r01-metric-aggregate \
  python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_alpha_r01_c5000_metric_campaign_v1_1788924371071068000_aggregate.json \
  --overwrite
```

Replace `--overwrite` with `--validate-only` to exercise all registered-input,
result-hash, path-hash, formal-completion, mesh, inner-cap, dense-convergence,
NoF-accounting, and reference-undershoot gates without writing outputs.  The
final validation found 8/8/8 runs in the three columns, 442 no-reparameterized
formal-complete paths, 11 of them common-mesh-eligible, and a nonbinding
\(m_{\max}=5000\) cap.  The final one-page PDF is 751.181 by 334.488 points;
all five fonts reported by `pdffonts` are embedded.  A 200-dpi Poppler render
was visually inspected after the final axis and color-map update.

## Presentation integration

The final PDF was copied without modification to
`/Users/wuzhiyou/Downloads/CiCP_StringNET_2026-3/figures/ivm/alanine/alanine66d_alpha_cost_to_precision_2x3.pdf`;
its SHA-256 remains
`bd988a1aeaeb273853a06cb5f006526429b75d96405427b5ab82abe67f0e078d`.
It is included on the standalone “Example 3.1: Ala Parameter Sensitivity”
frame in
`/Users/wuzhiyou/Downloads/CiCP_StringNET_2026-3/Personal_statement_wzy_short.tex`.
The final TeX SHA-256 is
`c2e414e090707f82a13e2573de4993ad945d67bd6e57578bac1fb3bcd0303c13`.
`latexmk Personal_statement_wzy_short.tex` completed successfully and wrote
the 43-page preview
`/Users/wuzhiyou/Downloads/CiCP_StringNET_2026-3/build/Personal_statement_wzy_short.pdf`
with SHA-256
`c3b9d4f1ffc600011446723a05d83e4044ada82af02fcc71fd4b4058b7f05a81`.
Page 29 was rendered at 200 dpi and visually checked: the figure and caption
are legible and there is no clipping, overlap, or missing content.
