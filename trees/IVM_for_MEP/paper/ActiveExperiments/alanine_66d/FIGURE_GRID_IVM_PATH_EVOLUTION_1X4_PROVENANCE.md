# Alanine 66D registered-grid IVM path-evolution figure

## Registered artifact

This record traces the manuscript Fig. 5 replacement built from the same
selected IVM run and the same frozen ODE--NEB baseline used in manuscript
Fig. 6.  It supersedes the legacy `fig_1x4_contours_custom.pdf`, whose IVM
trajectory came from an older experiment.

| role | absolute path | SHA-256 |
| --- | --- | --- |
| registered PDF | `/Users/wuzhiyou/IVM_for_MEP/output/pdf/alanine66d_grid_ivm_path_evolution_1x4.pdf` | `4175ba9eefd53905ec7ad81b6c85ced6e30d8c5cf2d1240cb4a48aafc2c5d3eb` |
| plotted coordinates | `/Users/wuzhiyou/IVM_for_MEP/output/pdf/alanine66d_grid_ivm_path_evolution_1x4_source.csv` | `f5b11fcf686cdeef60cae70124aa2ca375b05f3660df7575db11bf59063c9c95` |
| machine manifest | `/Users/wuzhiyou/IVM_for_MEP/output/pdf/alanine66d_grid_ivm_path_evolution_1x4_manifest.json` | `140898f6685e6b1204034c8cdd6a79a03cd371581d0f0914b59ab34e10003936` |
| plotter | `/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_grid_ivm_path_evolution_1x4.py` | `b5252cd9ce40a8ec23ae9add9dcc8e9cdeb5e75ba6feed26950c16ea77808313` |

The manuscript keeps byte-identical copies of the PDF, CSV, manifest, and
this provenance record under
`/Users/wuzhiyou/Downloads/IVM_for_MEP/paper0819/Figures/ala2_66d/`.

## Experiment identity

The IVM input is the registered `inexact_reparam` result with

```text
alpha = 10
r = 0.7
m_max = 5
total internal-image NoF budget = 5000
cell_id = a736dd230285d61c
```

Its frozen result is

`/Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801/runs/stage4_ala_discrete_inexact_reparam_grid_v1/members/ivm_stage4_ala_discrete_inexact_reparam_grid_v1_a1e1_rr70_c05_tnf5000_r1_central_state_ivm_cfgc3a418a9c878_budget5000.npz`

with SHA-256
`0da1bbab469de81ca5b9229c4ebef656b7688ccb79c822ec14f0f4c92226d161`.
The ODE--NEB input is the frozen `k=50` result
`ala_0to1_N22_k50_alpha0.0001_ode_results.npz`, SHA-256
`935561709ecbdfe87af31b78b45335a66545356b3e62f331a4a9fbb8125d9a09`.
The selected IVM identity is checked against the registered grid-metric JSON,
and the plotter fails closed if the controls, result hash, state hashes, or
cost ledgers change.

## Four panels and NoF selection

Panel a shows the selected IVM run's IDPP path before charged optimization
evaluations.  Panels b and c use the latest saved path of each method at or
below common budgets of 100 and 1000 force evaluations.  Panel d shows each
method's final reported path: the last formal complete, mesh-eligible IVM
state and the terminal saved ODE--NEB path.  Paths are never interpolated.

| panel | meaning | IVM record / actual NoF | ODE--NEB record / actual NoF |
| --- | --- | ---: | ---: |
| a | IVM IDPP initialization | initial / 0 | not plotted |
| b | latest paths with `N_F <= 100` | 4 / 100 | 2 / 82 |
| c | latest paths with `N_F <= 1000` | 52 / 993 | 41 / 982 |
| d | final reported paths | 235 / 3587 | 4637 / 100022 |

Panels b and c are trajectory snapshots, so their literal latest saved IVM
states are shown even though they are between committed reparameterized outer
states.  The `N_F=100` state satisfies the mesh thresholds but is not a formal
complete state; the `N_F=993` state is neither formal complete nor
mesh-eligible.  They are not used as Fig. 6 metric records, and these flags
are carried explicitly in the CSV and manifest.

The raw run continues to the nominal 5000 budget, but its tail is a partial,
uncommitted state that fails the registered mesh criterion.  Therefore panel
d uses the IVM's last formally complete path at `N_F=3587`, not the
quarantined 5000-NoF tail.  The ODE--NEB entry is its terminal saved path at
`N_F=100022`; neither path is labeled as convergence-certified.  The CSV has
154 data rows: 22 initial images and two 22-image method paths in each of
panels b--d.  Every row records the panel budget, actual NoF, source
record/call, image index, projected coordinates, full path hash, source file,
and source file hash.

The IVM optimization cost is `total_internal_joint_evaluations` over the 20
movable internal images.  ODE--NEB uses the frozen convention
`N_F = 20 * count_seq + 2`.  Offline projections and background loading add
no optimization NoF.

The ODE--NEB artifact's initial path is a numerically close legacy IDPP path,
not byte-identical to the selected IVM common-initial-path artifact (maximum
Cartesian difference `0.06150258892739646` in the stored coordinates).  Panel
a is therefore explicitly the selected IVM initialization and is not labeled
as a shared IVM/ODE initialization.

## Projection and visual policy

All optimization is in 66-dimensional Cartesian space.  The figure only
projects each saved 22-image path onto the Ramachandran dihedrals using the
same existing article helper as the legacy Ala figure:

```text
phi atoms: (4, 6, 8, 14)
psi atoms: (6, 8, 14, 16)
wrap: [-180, 180) degrees
```

The potential-energy background is the frozen `ala2_scan_cache.pkl`, shifted
by its minimum and converted with `23.0605 kcal mol^-1 eV^-1`.  Its `plasma`
palette, 0--15 kcal/mol filled range, white contour lines, and the original
blue/orange/gray method colors are retained.

Trajectory markers follow manuscript Fig. 4: IVM is a hollow circle and
ODE--NEB is a hollow square, at every one of the 22 images.  The initial path
uses a hollow gray diamond.  **No special endpoint or minimum markers are
drawn; in particular, the two former yellow stars have been removed.**

## Rebuild and verification

From the source repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/ala_grid_path_mpl \
XDG_CACHE_HOME=/private/tmp/ala_grid_path_cache \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_grid_ivm_path_evolution_1x4.py \
--overwrite
```

The manuscript preview is built without replacing formal `main.pdf`:

```bash
cd /Users/wuzhiyou/Downloads/IVM_for_MEP/paper0819
/Library/TeX/texbin/latexmk -pdf -synctex=1 -interaction=nonstopmode \
-file-line-error -halt-on-error -outdir=tmp/vscode-latex main.tex
```
