# Provenance: `Figures/LEPS/leps_all_methods_metrics.pdf`

**Current figure update (2026-09-10):** gMAM and the user-defined two-stage
StringNET have been added; `Original String` is displayed as `String Method`.
The six trajectories documented below are unchanged. The authoritative
current figure/data/plotting registry is now
`../leps_action_baselines_20260910/reports/leps_all_methods_metrics_provenance.json`;
see `../leps_action_baselines_20260910/README.md` for its full experiment and
NoF provenance. The PDF identities and six-method layout below describe the
historical 2026-09-06 version, preserved in the new campaign's archive.

## Registered figure and method name

On 2026-08-30 the LEPS all-methods figure was permanently updated by explicit
user authorization.  The selected IR-IVM trajectory now occupies the single
method slot named **IVM**.  In this figure and in subsequent LEPS discussion,
`IVM` means the refined-grid IR-IVM described below.  The former article IVM
trajectory is not plotted.  The five non-IVM comparison trajectories are
unchanged.

On 2026-09-06 the user authorized adding the discrete geometric action as
panel c and making all three plotting boxes square. Panels a and b retain the
same metrics, paths, NoF values, and presentation rules as the preceding
two-panel version.

The formal paper-working copy and the LEPS source copy are byte-identical:

| Role | Path | Bytes | SHA-256 |
|---|---|---:|---|
| Formal figure | `Figures/LEPS/leps_all_methods_metrics.pdf` | 84,614 | `081aad16011737c67d8d9eaded57559655f9b45f6ab18817940ca019c17f66b4` |
| LEPS source copy | `code_on_server/LEPS/figures/leps_all_methods_metrics.pdf` | 84,614 | `081aad16011737c67d8d9eaded57559655f9b45f6ab18817940ca019c17f66b4` |
| Output copy | `output/pdf/leps_all_methods_metrics.pdf` | 84,614 | `081aad16011737c67d8d9eaded57559655f9b45f6ab18817940ca019c17f66b4` |

The immediately superseded two-panel PDF had SHA-256
`9faaa4e643cd14329bdb65825f954886bb0232db04b0a425b7a7ae14a9051a25`;
the still earlier PDF had SHA-256
`f558d0b0aeb66115bc3b36a015aad2cc2bfce748917b7e309abc19f6589b0376`.
These identities remain recorded for historical audit but no longer identify
the file at the formal figure path.

## IVM experiment represented by the blue curve

The plotted IVM result is

`leps_irivm_sync1_refined_grid_20260830/runs/3c62d93a051be116.npz`

with SHA-256
`c7a09d0ae37cea45078dff2590e244390420c9b80931de1e4767e8b62f1bb02d`.
Its registered controls are

\[
\alpha=100,\qquad r=3\times10^{-3},\qquad m_{\max}=5.
\]

The raw result contains the common initial state and 28 completed outer
iterations.  All 29 accepted states appear in the plotted-point CSV and all
29 are shown with individual blue circular markers.  The run terminates at
1,558 Number of Forces; the last accepted path is at 1,528 Number of Forces.
Its exact optimization ledger is

\[
20\ \text{initial}+428\ \text{main}
+555\ \text{direct reparameterization}
+555\ \text{single recovery}=1558.
\]

Every completed outer iteration attempts one improved-tangent identical-spring
reparameterization and one synchronized recovery.  There are 28 such
transactions.  All 28 roll back, and all direct and recovery force calls remain
charged.  Rejected transaction states do not appear as accepted curve points.

The result is the fixed-rule winner of the preregistered 150-combination
refined Cartesian search and also of the deduplicated 186 combinations across
both registered search stages.  It satisfies both strict same-NoF inequalities
at 19 of 21 checks; the densified-barrier inequality misses at NoF 150 and 278.
No setting passed all 21 checks.  This status is retained here even though the
figure replacement was subsequently authorized.

## Experiment and algorithm source chain

The complete path from experiment definition to raw result is:

```text
registrations/CAMPAIGN_PREREGISTRATION.json
    + PROTOCOL.md
    + src/leps_irivm_sync1_refined.py
    + ../leps_irivm_ala_chig_every1_sync1_20260830/src/leps_irivm_sync1.py
    + inputs/common_initial_path.npz
    + src/leps_core.py
    -> src/run_grid.py
    -> runs/3c62d93a051be116.npz
    -> src/audit_and_select.py
    -> reports/acceptance/selection_manifest.json
```

The important registered artifacts are:

| Artifact | Purpose | SHA-256 |
|---|---|---|
| `registrations/CAMPAIGN_PREREGISTRATION.json` | frozen refined grid, method contract, metrics, and selection rule | `947cd467cf73ce18c230e951099f7406c6cd29e1c186dc2f17acc7e5439ce2f0` |
| `PROTOCOL.md` | human-readable experimental protocol | `3e47685f44e2b52238c8a30abe04419186d5d9c88f88226ea8469bd5843aa235` |
| `src/leps_irivm_sync1_refined.py` | frozen refined-grid wrapper | `721fb2b77d6711dc032f0578ec12651064711c2adbe8afcc839228120148e971` |
| `../leps_irivm_ala_chig_every1_sync1_20260830/src/leps_irivm_sync1.py` | IR-IVM optimizer and single synchronized recovery | `8b1f848d6a2f03090b4851b6bd4a45d55a61fbee4025dfa50f5e59e2084ea49a` |
| `src/run_grid.py` | serial, host-locked grid runner | `405de5e8c7096e586c976c9e91077c157a24b967b1704cd66467c2e3d6aa3072` |
| `src/audit_and_select.py` | independent metric, NoF, and selection audit | `683e02261e4731537b41343efbc0443f2417f4ddff883deeba08bd368a9ac436` |
| `reports/acceptance/selection_manifest.json` | refined-stage selection | `4ca04977a7331a93290a8f1b272b58213ef3f8c0bb502fa64d3c2611f44a1094` |
| `reports/acceptance/combined_selection_manifest.json` | deduplicated two-stage selection | `51e1979aa580afb1381cbe71573fa36f7db8cd04e208ed3097654b176ad53643` |
| `runs/3c62d93a051be116.npz` | selected raw optimization result | `c7a09d0ae37cea45078dff2590e244390420c9b80931de1e4767e8b62f1bb02d` |

The optimizer uses the complete-normal proximal Armijo L-BFGS main problem,
Henkelman-Jonsson improved tangents everywhere, the deterministic trust radius
`min(2 * median segment RMS length, 1)`, improved-tangent identical-spring
reparameterization after every outer iteration, and at most one synchronized
complete-band recovery evaluation.  Recovery uses `alpha` but does not read the
main-subproblem `r`.  It never reads the old article IVM trajectory, reference
saddle barrier, densified barrier, or figure-selection result.

## Metrics and Number of Forces

For every accepted path shown in the figure:

- `I_perp` is recomputed using the article path-length quadrature and the
  Henkelman-Jonsson improved tangent;
- the plotted energy barrier is the maximum energy on 500 total points from
  coordinate-wise linear interpolation of the path;
- the geometric action is the 22-node segment-length trapezoidal integral
  `sum_i 0.5*(|grad V_i|+|grad V_(i+1)|)*|x_(i+1)-x_i|`, including both
  endpoints and using the full gradient norm without densification;
- this action is tangent-independent; the unchanged Henkelman-Jonsson
  improved tangent remains the tangent convention for `I_perp`;
- Number of Forces is the total number of uncached joint energy-force
  evaluations over the 20 internal images;
- direct reparameterization, recovery, rejected trials, and rolled-back
  transactions are included in Number of Forces;
- separate 500-point metric evaluations are diagnostic calculations and are
  not added to optimization Number of Forces.

## Other five plotted trajectories

The following frozen production results supply the non-IVM curves:

| Method | Raw result | SHA-256 |
|---|---|---|
| ODE-NEB | `runs/neb/neb_production_k10_cfgfe491f5a6c6f_budget5000.npz` | `ae0c2bbf3f8fa8539bcaffaa98e0298b5f90661f145fcb5e7a5c6b61c4548b0a` |
| Static-NEB | `runs/neb/production_static_k10_dt0.05_cfgba1c63497f9a_budget3000.npz` | `a8daa64a59d7316c4546f47591cbfcdd509c3744eb0def2390c3951c5cdcb046` |
| Original String | `runs/original_string/production_original_dt0.07_r1_cfg6446cefea5fd_budget3000.npz` | `c480e8520ff37b493e20c8dd476655aee58d0ebbd119360eccd74f8971f1a4cd` |
| OBSM-FIRE | `runs/obsm_fire/production_fire_stable_l0.85_n4_dt0.18_cfgc5d07bd3b321_budget3000.npz` | `24dc2c8882d3cab39861fb69a8630b317532f26a79bd6c7fb913a1ce23faa456` |
| OBSM-L-BFGS | `runs/obsm_lbfgs/production_lbfgs_r5_l0.95_n1_s0.1_cfg78c1b29f7570_budget3000.npz` | `b5ae71ab114e8dffb90be582b327c1c6468f13123289b4358cdd3ea143be22cb` |

All paths in this table are relative to `code_on_server/LEPS`.

## Figure construction and machine-readable plotted data

The current three-panel figure is generated by
`../leps_geometric_action_20260904/src/remake_leps_all_methods_three_metrics.py`,
SHA-256
`107942f1ece049e223fa47ca9536eacbeb6ab326d51a5b9db30ec1dd0af202fe`.
It imports the registered trace readers and visual identities from
`src/remake_leps_all_methods_selected.py` without calling that script's
two-panel `main` routine. It reuses the frozen metric primitives in
`../ir_ivm_qm10_v2p20_microrefine_20260829/src/remake_leps_all_methods.py`,
SHA-256
`8ee8b750f899b886a80ad2aeab0b52ec9239f5e425301f35f12af9c3bb67cf2e`.

The exact plotted points for all three metrics are in
`reports/figure/leps_all_methods_metrics_source.csv`, SHA-256
`5fbccdaba1ca0b8612ff3860b52cadea4be0bf0aa7b42889e6b8ad95a3712138`.
It records method name, plotted index, Number of Forces, all three recomputed
metrics, quadrature details, raw-result path, and raw-result SHA-256. It contains exactly six
method names: `IVM`, `ODE-NEB`, `Static-NEB`, `Original String`, `OBSM-FIRE`,
and `OBSM-L-BFGS`; there is no `Article IVM` row.

The machine-readable reconstruction manifest is
`reports/figure/leps_all_methods_metrics_provenance.json`. It records the
current figure/script/data hashes, three metric contracts, 1x3 arrangement,
and verified display-width/display-height ratios for every panel.

Rebuild and verify the registered figure with:

```bash
cd /Users/wuzhiyou/Downloads/IVM_for_MEP

MPLCONFIGDIR=/private/tmp/leps-three-metrics-mpl \
  /Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  code_on_server/LEPS/leps_geometric_action_20260904/src/remake_leps_all_methods_three_metrics.py
```

The one-page PDF was rendered with Poppler after replacement and checked for
legend content, curve count, clipped labels, overlaps, marker visibility, and
panel alignment. The three plotting boxes have verified width/height ratio
exactly 1 to floating-point precision. The legend contains `IVM` and does not
contain `IR-IVM` or `Article IVM`; all 29 IVM accepted states are visibly marked.
