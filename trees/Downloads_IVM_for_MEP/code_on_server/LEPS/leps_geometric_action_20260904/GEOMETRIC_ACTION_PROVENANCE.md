# LEPS discrete geometric action: figure and data provenance

Registered on 2026-09-04. This is an additional diagnostic figure, not a new
optimization experiment or a new parameter selection. On 2026-09-06 the same
action values were incorporated as panel c of
`Figures/LEPS/leps_all_methods_metrics.pdf`, whose three plotting boxes were
made square. The superseded two-panel figure had SHA-256
`9faaa4e643cd14329bdb65825f954886bb0232db04b0a425b7a7ae14a9051a25`.

## Figure, computation, and exact plotted data

- Paper-working figure: `Figures/LEPS/leps_all_methods_geometric_action.pdf`
  (relative to the project root).
- Identical output copy: `output/pdf/leps_all_methods_geometric_action.pdf`.
- Calculation and plotting script: [src/plot_leps_geometric_action.py](src/plot_leps_geometric_action.py).
- Exact plotted values: [reports/leps_all_methods_geometric_action_source.csv](reports/leps_all_methods_geometric_action_source.csv).
- Per-path segment lengths, node gradient norms, improved tangents, NoF, and
  recomputed I_perp: [reports/quadrature_and_tangent_audit.npz](reports/quadrature_and_tangent_audit.npz).
- Numeric summary: [reports/summary.json](reports/summary.json).
- SHA-256 identities of the PDF, script, six raw results, original figure and
  source table, metric code, and new diagnostic outputs:
  [reports/provenance.json](reports/provenance.json).

## Definition and discretization

For the user's definition

\[
\widehat S(\varphi)=\int_\varphi\lVert\nabla V\rVert\,\mathrm d\ell,
\]

the discrete figure uses the original 22 nodes (20 internal images and two
fixed endpoints) and a trapezoidal rule on the 21 polygonal segments:

\[
\widehat S_h=\sum_{i=0}^{20}
\frac{\lVert g_i\rVert+\lVert g_{i+1}\rVert}{2}
\lVert x_{i+1}-x_i\rVert,
\qquad g_i=\nabla V(x_i).
\]

Both endpoint contributions are included; their gradients are evaluated, not
assumed to be zero. The equivalent node weights are half the adjacent segment
length at each endpoint and half the sum of the two adjacent segment lengths
at every internal image. No path densification or additional interpolation is
used for this integral. Values have units of eV.

Energy and analytic gradients come directly from `src/leps_core.py` under the
LEPS root, through `leps_energy_gradient`. Distances and norms use the same
flat six-coordinate Euclidean representation as the existing LEPS metric
code: only coordinates 0 and 3 vary, and the four inactive coordinates are
verified to be zero throughout every saved path.

### Tangent convention

The tangent implementation is exactly `leps_core.improved_tangents`, which
calls ASE's energy-weighted Henkelman-Jonsson `ImprovedTangentMethod`. No
central-difference tangent is introduced. We recompute the original I_perp
using this tangent and verify every result against the original figure's
plotted-point table.

The requested action itself uses the **full gradient norm**, so it does not
require a tangent projection. It is not an integral of normal force, and does
not subtract a gradient-tangent term or introduce an extra factor of 1/2.
The identity
`|g|^2 = |g_perp|^2 + (g dot tangent)^2` is independently checked at all
internal nodes with the same improved tangents.

This is a coarse, node-based quadrature, as requested. It is not the exact
continuous action: differently spaced discrete paths may have different
quadrature errors. A smaller value alone is not a certificate that I_perp or
the densified barrier is also smaller, nor a certificate of convergence.

## Trajectory and experiment provenance

The data source is the current registered six-method figure, not the former
article IVM trajectory. The authoritative upstream registration is
[FIGURE_PROVENANCE.md](../leps_irivm_sync1_refined_grid_20260830/FIGURE_PROVENANCE.md),
with its machine-readable manifest at
`leps_irivm_sync1_refined_grid_20260830/reports/figure/leps_all_methods_metrics_provenance.json`
under the LEPS root. The present script verifies the raw-result hashes from
that manifest before calculating the action.

| Legend name | Raw result, relative to `code_on_server/LEPS` |
|---|---|
| IVM | `leps_irivm_sync1_refined_grid_20260830/runs/3c62d93a051be116.npz` |
| ODE-NEB | `runs/neb/neb_production_k10_cfgfe491f5a6c6f_budget5000.npz` |
| Static-NEB | `runs/neb/production_static_k10_dt0.05_cfgba1c63497f9a_budget3000.npz` |
| Original String | `runs/original_string/production_original_dt0.07_r1_cfg6446cefea5fd_budget3000.npz` |
| OBSM-FIRE | `runs/obsm_fire/production_fire_stable_l0.85_n4_dt0.18_cfgc5d07bd3b321_budget3000.npz` |
| OBSM-L-BFGS | `runs/obsm_lbfgs/production_lbfgs_r5_l0.95_n1_s0.1_cfg78c1b29f7570_budget3000.npz` |

The IVM label retains its current meaning: the selected IR-IVM with
`alpha=100`, `r=3e-3`, and `m_max=5`. Its experiment wrapper is
`leps_irivm_sync1_refined_grid_20260830/src/leps_irivm_sync1_refined.py`; its
optimizer is
`leps_irivm_ala_chig_every1_sync1_20260830/src/leps_irivm_sync1.py`.
Their hashes, preregistration, protocol, runner, and selection audit are
linked in the upstream registration. No optimizer, recovery, tangent,
parameter, initialization, or accepted state is changed by this calculation.

The registered plot's trace readers, method colors, line styles, markers, and
NoF range are imported from
`leps_irivm_sync1_refined_grid_20260830/src/remake_leps_all_methods_selected.py`
and its frozen plotting primitives. Their `main` routines are **not** called,
so this standalone-action script cannot replace the registered three-panel
figure or its registry.

## Number of Forces and presentation

Every plotted method/state/NoF tuple is checked against the original
`leps_all_methods_metrics_source.csv`. The action figure has the same
0–20,000 NoF range and plots all 3,577 original states within that range.
It connects raw recorded values without smoothing, a best-so-far envelope,
or extrapolation beyond each method's last saved path.

For IVM, all 29 states (initial state and 28 completed outer iterations) have
individual blue markers. The final saved path is at NoF 1,528, while the
complete optimization ledger ends at 1,558; the extra rejected evaluations
do not create another accepted-path point. As in the original figure, the
optimization NoF includes rejected trials and recovery calls.

The new offline calculation evaluates analytic gradients at 78,694 nodes
(3,577 paths times 22 nodes). These are diagnostic evaluations, separately
recorded in the manifest, and are not added to optimization NoF. No force
budget or optimization result is altered.

## Numerical results

Every method has the same initial path and action `10.709383931470192` eV.
The following are last plotted states at **different** costs, not a same-NoF
ranking:

| Method | Saved paths | Last plotted NoF | Action at last plotted state (eV) |
|---|---:|---:|---:|
| IVM | 29 | 1,528 | 1.7951584608504483 |
| ODE-NEB | 930 | 20,000 | 1.8525728745462033 |
| Static-NEB | 1,000 | 20,000 | 7.501866583980564 |
| Original String | 1,000 | 20,000 | 23.129422083492397 |
| OBSM-FIRE | 200 | 19,920 | 1.662599932549530 |
| OBSM-L-BFGS | 418 | 19,985 | 7.621216295869827 |

IVM's smallest recorded value is `1.7927613143384924` eV at NoF 517; it is
slightly below its final value. The curve displays that nonmonotonicity rather
than replacing subsequent values with the historical minimum. OBSM-FIRE's
late value is lower than IVM's final value for this discrete diagnostic.

## Reproduction and verification

From the project root:

```bash
MPLCONFIGDIR=/private/tmp/leps-geometric-action-mpl \
  /Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  code_on_server/LEPS/leps_geometric_action_20260904/src/plot_leps_geometric_action.py
```

The script recreates only the new action PDF/copy and new reports. The
original PDF, registry, plotted-point table, and six raw inputs are checked
for unchanged hashes after completion. Runtime versions are in the new
manifest.

Checks completed:

- All six registered raw-result hashes and every original method/NoF tuple
  agree; all six initial paths are identical.
- Constant-gradient, reversal, zero-gradient, and endpoint-weight quadrature
  tests pass; segment and node-weight sums agree.
- All improved tangents are unit length to a maximum error of
  `2.220446049250313e-16`.
- Recomputed I_perp agrees with the original figure to a maximum absolute
  difference of `4.263256414560601e-14`.
- The standalone one-page PDF was rendered with Poppler and visually checked for labels,
  six-method legend, raw curve visibility, and clipping. The two new PDF
  copies are byte-identical. The later combined three-panel figure has its own
  current registration in
  `../leps_irivm_sync1_refined_grid_20260830/FIGURE_PROVENANCE.md`.
