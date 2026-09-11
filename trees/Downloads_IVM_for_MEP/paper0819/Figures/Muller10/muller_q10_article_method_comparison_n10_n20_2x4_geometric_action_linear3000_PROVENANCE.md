# Registered QM10 comparison without geometric action

This record identifies the formal QM10 figure, the plotting program, the
optimization outputs used by every curve, and the algorithm source that
generated the selected IVM trajectories. Exact absolute paths and SHA-256
digests are stored in the companion machine-readable manifest.

## Registered artifact

- PDF: `muller_q10_article_method_comparison_n10_n20_2x4_geometric_action_linear3000.pdf`
- PNG preview: `muller_q10_article_method_comparison_n10_n20_2x4_geometric_action_linear3000.png`
- Exported plotted values: `muller_q10_article_method_comparison_n10_n20_2x4_geometric_action_linear3000_source.csv`
- Manifest: `muller_q10_article_method_comparison_n10_n20_2x4_geometric_action_linear3000_manifest.json`
- Manuscript location: `../../main.tex`, section label
  `subsubsec:Muller10`, figure label `fig:M10_density`

The filename is retained for compatibility with the requested manuscript
path. Despite the historical `2x4_geometric_action` stem, the registered PDF
is a 2-by-3 figure and contains no geometric-action panel or data series.

## Exact reproduction command

Run from `/Users/wuzhiyou/Downloads/IVM_for_MEP/paper0819`:

```sh
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python Figures/Muller10/plot_muller_q10_article_method_comparison_n10_n20_2x3_no_action.py
```

The manifest records the executable and the Python, NumPy, and Matplotlib
versions observed during generation. The system `python3` is not the
registered interpreter for this artifact.

## Figure and code chain

The primary plotting source is
`plot_muller_q10_article_method_comparison_n10_n20_2x3_no_action.py`. Its
registered plotting dependencies include the IR-IVM overlay loader, the base
QM10 comparison loader, the formal notebook-derived plotting definitions, the
QM10 potential, and the improved-tangent implementation. The historical
geometric-action program is imported only as a layout/data-validation helper;
no geometric action is evaluated or exported by the no-action program.

The selected IVM curves were produced by the single registered optimizer
source
`/Users/wuzhiyou/IVM_for_MEP/python/q10_irivm_recovery_monotone_exploration_20260831/src/q10_horizon_control_probe_optimizer.py`.
The manifest also authenticates its tangent/secant core, run-and-score entry
point, and independent metric auditor. Both path resolutions record the same
optimizer SHA-256. Only the three principal controls differ:

- 10 interior images: `(alpha, r, m_max) = (3e-3, 0.1, 7)`;
- 20 interior images: `(alpha, r, m_max) = (1e-3, 0.3, 8)`.

## Input data chain

For each row and method, the CSV stores the source trajectory path, its
SHA-256, and a zero-based `source_record_index`. This index preserves source
order and disambiguates repeated Number-of-Forces values without merging
them. The manifest additionally lists all 12 plotted trajectory archives and
the two formal IVM summary JSON files with hashes. The formal summaries
authenticate the optimizer source used to generate the two IVM trajectories.

The sole series labeled `IVM` is the selected unified inexact-recovery IVM.
The article-era IVM and the previous recovery variant are excluded from both
rows, both contour panels, and the exported CSV.

## Numerical conventions

- `I_perp` is the archived path-integrated normal-force metric computed with
  the Henkelman--Jonsson improved tangent.
- `Delta E` is the raw barrier from 500 equally spaced points obtained by
  coordinate-wise linear interpolation in normalized image index. It is not
  the barrier error. The registered reference barrier is
  `67.50188060819488`.
- `I_perp` uses the registered compressed logarithmic ordinate; `Delta E`
  uses an ordinary logarithmic ordinate.
- Number of Forces is the direct sum of uncached force evaluations over the
  movable images. Fixed endpoints and diagnostic densification are excluded;
  rejected and recovery trials remain charged.
- Rows use 10 and 20 interior images. The path panels use the last actually
  paid checkpoint not exceeding 3000 Number of Forces for IVM, ODE-NEB, and
  OBSM-LBFGS.

## Validation gates

The generator compares every exported `I_perp` and `Delta E` value, in source
order, against the registered pre-removal curve table. Duplicate NoF records
must remain present, and the maximum differences for both metrics must be
zero. It also verifies the fixed transformation matrix and its orthogonality,
checks the formal IVM trajectory identities and algorithm hashes, and refuses
to register a manuscript section that does not include this PDF. The manifest
is marked `visual_qa_passed` only after the generated PDF and manuscript
preview have been rendered and inspected.
