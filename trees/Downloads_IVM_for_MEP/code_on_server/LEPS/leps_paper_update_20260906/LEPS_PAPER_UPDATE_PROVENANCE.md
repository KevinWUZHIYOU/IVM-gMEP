# `paper0819` LEPS figure and prose update

This registry records the 2026-09-06 replacement of all three LEPS figures
and the corresponding LEPS prose in `paper0819/main.tex`. No optimizer was
rerun. Every curve and path was reconstructed from frozen production results;
dense evaluations are offline diagnostics and are excluded from Number of
Forces (NoF).

## Current IVM definition

In all three figures, `IVM` denotes the selected IR-IVM with

\[
\alpha=100,\qquad r=3\times10^{-3},\qquad m_{\max}=5.
\]

Its raw result is
`code_on_server/LEPS/leps_irivm_sync1_refined_grid_20260830/runs/3c62d93a051be116.npz`.
The complete experiment, algorithm, grid-search, and NoF provenance remains in
`code_on_server/LEPS/leps_irivm_sync1_refined_grid_20260830/FIGURE_PROVENANCE.md`.

## Three paper figures

| Figure used by `paper0819/main.tex` | Construction |
|---|---|
| `paper0819/Figures/LEPS/leps_path_evolution_1x4.pdf` | Recomputed from the selected IVM and registered ODE-NEB raw path histories. Panel a is the byte-identical common IDPP path; panels b-c use the latest saved state at or below 200 and 1500 total NoF, and panel d compares each method's final saved path. |
| `paper0819/Figures/LEPS/leps_all_methods_metrics.pdf` | Byte-identical copy of the current registered three-panel figure in `Figures/LEPS/`; panels are discrete `I_perp`, 500-point densified barrier, and 22-node full-gradient geometric action. |
| `paper0819/Figures/LEPS/leps_continuous_methods_comparison.pdf` | Recomputed from selected IVM, ODE-NEB, NN-IVM, and gMAM frozen results. All curves use total internal-image/stream NoF; discrete dense metrics use 200 linear subdivisions per segment. |

All path panels and all metric panels have a verified square plotting box.
The exact current PDF hashes, raw-input hashes, source-table paths, figure
construction rules, selected snapshots, and numerical summaries are in
`reports/provenance.json`.

The path-snapshot coordinate table is
`reports/leps_path_evolution_source.csv`; it records budget, actual NoF,
method, image index, and both LEPS coordinates. The continuous-figure plotted
points are in `reports/leps_continuous_methods_source.csv`. The exact
all-method plotted points remain in
`../leps_irivm_sync1_refined_grid_20260830/reports/figure/leps_all_methods_metrics_source.csv`.

## Data-backed prose values

- At nominal path budgets 200/1500 NoF, the selected IVM states are at
  135/1486 NoF and the ODE-NEB states are at 200/1500 NoF. The final-path
  panel uses 1528 NoF for IVM and 100000 NoF for ODE-NEB.
- IVM first reaches discrete `I_perp <= 1e-4` at 572 NoF; ODE-NEB first does
  so at 9940 NoF. The other four all-method baselines do not reach it by
  20,000 NoF.
- The last accepted IVM path is at 1528 NoF with
  `I_perp = 5.571619177163831e-28`; the complete ledger stops at 1558 NoF.
- IVM's discrete geometric action decreases from 10.709383931470192 eV to
  1.7951584608504483 eV; its minimum saved value is 1.7927613143384924 eV at
  517 NoF. OBSM-FIRE reaches 1.66259993254953 eV at 19,920 NoF but retains
  `I_perp = 0.5065056474227418`, so action is not used alone as an MEP test.
- In the dense representation comparison, IVM reaches minimum dense
  `I_perp = 0.8065852288698554` and minimum barrier error
  `0.001104686603424998` eV at 517 NoF. ODE-NEB reaches minimum dense
  `I_perp = 0.9192921145049106` at 100,000 NoF. NN-IVM reaches minimum dense
  `I_perp = 0.10082256737259543` at 9820 NoF and minimum barrier error
  `0.00022665984811620632` eV at 4400 NoF. gMAM reaches minimum dense
  `I_perp = 0.007945648888769615` at 5780 NoF on the high-barrier branch.

## Rebuild

From the project root:

```bash
MPLCONFIGDIR=/private/tmp/leps-paper-update-mpl \
  /Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  code_on_server/LEPS/leps_paper_update_20260906/src/update_paper_leps_figures.py

cd paper0819
/Library/TeX/texbin/latexmk -pdf -synctex=1 -interaction=nonstopmode \
  -file-line-error -halt-on-error -outdir=tmp/vscode-latex main.tex
```

The second command writes only the preview build under
`paper0819/tmp/vscode-latex/`; the formal `paper0819/main.pdf` is not
overwritten.
