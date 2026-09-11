# Alanine 66D fixed-alpha 2x2 cost-to-precision figure provenance

## Registration status

**Complete and visually verified.**  Both registered metric inputs pass the
plotter's read-only validation.  The PDF, the exact plotted-record table, and
the machine-readable manifest below were generated from those frozen inputs;
the historical alpha figures remain unchanged.

Stable stem:

`alanine66d_alpha_cost_to_precision_2x2`

Registered outputs:

| Artifact | Location | SHA-256 |
|---|---|---|
| PDF | `output/pdf/alanine66d_alpha_cost_to_precision_2x2.pdf` | `2dad0762cc8aa0f2f6a6022620a5ea6ab84a0f517386603c8be4da6ed08f1fa5` |
| Plotted-record table | `output/pdf/alanine66d_alpha_cost_to_precision_2x2_source.csv` | `5d6d21435e7876fe77ea17f63dc3eeaa7893cf05f46ea29188dbbf10628ee835` |
| Machine-readable manifest | `output/pdf/alanine66d_alpha_cost_to_precision_2x2_manifest.json` | `2c5bd367a13b421e25d575d49c72c93d31f75a19b2a83ce887a23bab306fa913` |
| Plotter | `paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_alpha_cost_to_precision_2x2.py` | `6d379347492872d4ef68f92b5b4fbcc46f9511add285adb73542fa42a8885b28` |

## Scientific comparison held by the plotter

The two columns use the same reparameterized Ala IVM implementation, the same
common IDPP initial path, the same fixed-
\(\alpha\) grid \(10^{-1},10^0,\ldots,10^6\), and the same budget of 5000
force evaluations at the moving images.  The only difference is the
accuracy allowed for each inner subminimization:

| Column | Inner-solve controls | Reparameterization |
|---|---|---|
| left | \(r=0.1\), implementation safety limit \(m_{\max}=5000\), required to be nonbinding | unchanged, enabled after every outer solve |
| right | production setting \(r=0.7\), \(m_{\max}=5\) | unchanged, enabled after every outer solve |

The left column must not be generated if any registered calculation reports
an `evaluated-inner-max-steps` stop.  The older \(r=0.1,m_{\max}=50\)
calculations do not satisfy this condition at high \(\alpha\), and the
existing exact/no-reparameterization calculations change a second algorithmic
component.  Neither is an admissible substitute for the new left column.

The right column is frozen to:

| Input | Location | SHA-256 |
|---|---|---|
| metric preregistration | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_grid_metric_campaign_v2_1787608620070816000_preregistered.json` | `5fcfad3228e72c267ea1d768dd6f677528ae1fc8e8c6f180ea0d3498503b98a3` |
| completed metric aggregate | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_grid_metric_campaign_v2_1787608620070816000_aggregate.json` | `0e53a4e00a8a30f8e60c9fd278426f9b55fecada08840dee6f514a4e93390e63` |

The completed left metric inputs are:

| Tighter-inner metric input | Location | SHA-256 |
|---|---|---|
| preregistration | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_alpha_r01_c5000_metric_campaign_v1_1788924371071068000_preregistered.json` | `540b6a985d09219cb112093003ce7f63de709cd3cb071a9f9e78645a2dd201a4` |
| completed aggregate | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_alpha_r01_c5000_metric_campaign_v1_1788924371071068000_aggregate.json` | `0f7a5c9220dace0468e0f1edd56a4693de71526d9b7f1682ad5ddc3f384a2295` |
| local task report | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_alpha_r01_c5000_metric_campaign_v1_1788924371071068000_local_run_report.json` | `f5f7fdf368faec673576ac2cb36f83f84d6a7059fa814d1f1f91897e8b2e0d22` |

These paths are supplied explicitly through the plotter CLI.  Their hashes,
all eight frozen optimizer-result hashes, every referenced dense-metric task
artifact hash, and the registered source-code hashes are recorded in the final
manifest.

The eight new optimizer calculations and their dense metrics have completed;
the optimizer calculations are frozen separately as follows:

| Tighter-inner optimizer input | Location | SHA-256 |
|---|---|---|
| preregistration | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_campaign_v1_1788923892307558000_preregistered.json` | `da1ded53f6dc11ac71ebe5bae64ba452f19c95fdc60467be220ff630912b6a05` |
| completed aggregate | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_campaign_v1_1788923892307558000_aggregate.json` | `50914e1cdfdafc59287921b4eef6081a5274d40366a4da926a590c06a6c9828a` |
| local worker report | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_campaign_v1_1788923892307558000_local_report.json` | `78ef8b82b7b7a8dbc2b6f4f62d8728fb216c1494fce46a95bbddce5241592362` |

The aggregate records eight unique completed calculations and certifies that
the inner-iteration safety limit was nonbinding in all eight.  The plotter
freezes these three identities and requires every new left metric record to
point back to the corresponding optimizer result and SHA-256.

## Saved-path and cost rules

Only paths fixed by the metric preregistration are admissible:

- the initial path or a complete, reparameterization-committed path;
- common mesh criteria
  \(\operatorname{CV}\le 0.0065\) and
  \(\max\Delta s_i/\min\Delta s_i\le1.03\);
- no terminal incomplete tail.

The ordinate is `total_internal_joint_evaluations`.  The fixed endpoints and
the independent densified-path diagnostics are excluded.  At a repeated
observed Number of Forces, the lowest actual metric value is retained.  The
curve then connects only strict improvements in increasing force cost.  No
metric or cost is interpolated, no curve is extrapolated, and no value is
carried forward to the 5000-force budget.

The rows are:

1. raw \(I_\perp\);
2. the actual densified energy barrier \(\Delta E\), not its reference error.

Each densified barrier is evaluated on the registered nested Cartesian grids
with 20, 40, 80, and 160 subdivisions per original segment.  The 160-level
barrier is selected, and the path must satisfy
\(|\Delta E_{160}-\Delta E_{80}|\le10^{-3}\)
\(\mathrm{kcal\,mol^{-1}}\).  These offline evaluations are not added to the
optimization cost.

The lower panels mark
\(\Delta E_{\mathrm{ref}}=2.4560825131253083\,
\mathrm{kcal\,mol^{-1}}\).  If any eligible new left-column path lies below
this reference by more than \(10^{-12}\,\mathrm{kcal\,mol^{-1}}\), the
plotter fails.  In that event a scientifically meaningful cost-to-precision
plot must instead use and explicitly label
\(|\Delta E-\Delta E_{\mathrm{ref}}|\); it must not silently retain raw
\(\Delta E\).

## Panel language, display windows, and common axes

- **(a), (c):** “Tighter inner solves with reparameterization
  (\(r=0.1\), \(m_{\max}=5000\); cap nonbinding)”.
- **(b), (d):** “Production inner solves with reparameterization
  (\(r=0.7\), \(m_{\max}=5\))”.
- top row: `Raw I_perp (smaller is better)`;
- bottom row: `Densified Delta E (kcal mol^-1)`;
- ordinate: `Number of Forces`.

The two columns share identical axes within each row, and all four panels use
the same logarithmic Number-of-Forces range.  The eight values of \(\alpha\)
share one color-and-marker legend.  Increasing \(\alpha\) follows the discrete
`jet` color map from blue to red.  The displayed windows are
\(I_\perp:10^3\to10^{-4}\),
\(\Delta E:4.0\to2.44\,\mathrm{kcal\,mol^{-1}}\), and
Number of Forces (18\to5500\).  The wider \(I_\perp\) window deliberately
retains the early stagnation of the tighter-inner calculations at large
\(\alpha\).  The source CSV retains every strict observed record, including
records outside the displayed energy-barrier window.

## Registered build command

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/ala_alpha_ctp_mpl \
XDG_CACHE_HOME=/private/tmp/ala_alpha_ctp_cache \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_alpha_cost_to_precision_2x2.py \
  --left-metric-preregistration \
  python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_alpha_r01_c5000_metric_campaign_v1_1788924371071068000_preregistered.json \
  --left-metric-aggregate \
  python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_alpha_r01_c5000_metric_campaign_v1_1788924371071068000_aggregate.json \
  --overwrite
```

Run first with `--validate-only` to exercise every identity, mesh, cap,
dense-convergence, and reference-undershoot gate without writing outputs.  The
registered PDF was rendered at 200 dpi with Poppler and inspected after the
final `jet` color-map and horizontal-window update.
