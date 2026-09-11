# Chignolin cost-to-precision design — revision 20260727c

## Curve construction

Each displayed curve is the piecewise-linear connection of the measured
first-passage Pareto vertices:

1. order the saved records by the total number of force evaluations;
2. retain an observation only when it establishes a new best value of the
   plotted metric;
3. plot that attained metric value on the horizontal axis and its total
   whole-band force cost on the vertical axis;
4. join successive record-setting observations by straight line segments.

This removes the artificial staircase produced by evaluating a dense grid of
thresholds. It does not smooth, fit, or extrapolate the measurements. The CSV
export of the dense threshold first-passage calculation is retained separately.

## Black speedup brackets

The black vertical bracket in each panel compares IVM and ODE-NEB at the
strictest threshold reached by both methods.

| Metric | Shared threshold | IVM forces | ODE-NEB forces | IVM speedup |
|---|---:|---:|---:|---:|
| Unnormalized `I_perp` | 0.4642178187 | 802 | 10002 | 12.47× |
| Densified barrier (kcal/mol) | 25.4902651151 | 902 | 10002 | 11.09× |

The annotation values are exported in
`data/chignolin_cost_to_precision_annotations_total_nf.csv`.

## Visual hierarchy

- IVM remains blue (`#0072B2`) and is the dominant line.
- ODE-NEB remains orange (`#D55E00`) and is the primary baseline.
- Static-NEB, Original String, and OBSM-FIRE use muted secondary colors and
  distinct line styles.
- The black brackets encode only the same-precision cost ratio and are not an
  additional method.

## Files

- Generator SHA-256:
  `8c509945c9d215b349a51fd68b3b7613048357510f481eff5e672863f2c6ed1e`
- PDF SHA-256:
  `abee190a71b2e3e3362ea025024f108a01ec1a4bde9db9173066b1d352f7fc4c`
- Output directory:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/figures/paper_methods_total_nf_20260727c`

The PDF is one page with all fonts embedded. The PDF, SVG, and 600-dpi PNG
representations are generated from the same Matplotlib figure.
