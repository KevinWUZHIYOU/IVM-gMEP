# Chignolin three-figure article audit

Date: 2026-07-27

## Article-ready directory

`/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/figures/article_update_20260727d`

The three recommended article figures are:

1. `chignolin_main_metrics_total_nf.pdf`
2. `chignolin_cost_to_precision_total_nf.pdf`
3. `chignolin_ivm_fixed_alpha_sensitivity_total_nf.pdf`

PDF, SVG, and 600-dpi PNG variants are generated from the same sources.

## Review of the previous 20260727a figures

### Main metrics

The data and total-force accounting were correct. The visible defect was that
panel label `b` collided with the upper part of the vertical barrier-axis
unit. The revised version moves every panel letter directly above the left
axis spine and removes the collision.

The horizontal axis is

```text
counter_vectors.sum(axis=1)
```

and therefore counts joint energy-force cache misses over all 22 images,
including the fixed endpoints' first evaluations.

### Fixed-alpha sensitivity

The data and budgets were correct. The previous legend used
`N_F <= 1,002`, `N_F <= 3,002`, and `N_F <= 6,002`. Inspection of the exported
checkpoint table shows that every alpha run is evaluated at exactly 1,002,
3,002, and 6,002 total force evaluations. The revised legend therefore uses
the more precise equalities

```text
N_F = 1,002, 3,002, 6,002.
```

The panel-letter placement is also made consistent with the other two
figures. Both axes retain logarithmic scaling to match the established
alanine fixed-alpha sensitivity figure.

### Cost-to-precision

The article candidate uses the corrected convention:

- horizontal axis: target precision threshold;
- vertical axis: minimum total Number of Force Evaluations;
- connected measured Pareto-front vertices rather than an artificial
  threshold-grid staircase;
- linear vertical axis;
- black double-headed arrows for IVM speedup relative to ODE-NEB and
  OBSM-FIRE at the strictest threshold attained by all three methods.

No unattained threshold is extrapolated.

## Common scientific and visual contract

- Same IDPP initial path hash:
  `57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996`.
- Same normalized central-difference tangent for every method.
- IVM color: `#0072B2`.
- ODE-NEB color: `#D55E00`.
- Static-NEB, Original String, and OBSM-FIRE are distinguished by both color
  and line style.
- Barrier-densification diagnostic calls are excluded from the optimization
  Number of Force Evaluations and reported separately.

## Cost-arrow values

| Metric | Shared threshold | IVM cost | Baseline | Baseline cost | Speedup |
|---|---:|---:|---|---:|---:|
| `I_perp` | 0.4642178 | 802 | ODE-NEB | 10002 | 12.5x |
| `I_perp` | 0.4642178 | 802 | OBSM-FIRE | 9482 | 11.8x |
| Dense barrier | 25.4903 kcal/mol | 902 | ODE-NEB | 10002 | 11.1x |
| Dense barrier | 25.4903 kcal/mol | 902 | OBSM-FIRE | 7062 | 7.8x |

## File validation

| Figure | PDF pages | Unembedded fonts | PNG dimensions |
|---|---:|---:|---:|
| Main metrics | 1 | 0 | 4356 x 1711 |
| Fixed-alpha sensitivity | 1 | 0 | 4277 x 1738 |
| Cost-to-precision | 1 | 0 | 4546 x 1702 |

All generated SVG files parse successfully. The final rendered PDFs have no
clipping, overlapping text, missing glyphs, or black-box artifacts. The full
53-test Chignolin suite passes.

## SHA-256

- Main metrics PDF:
  `53d3d26b9711ae04c5a08c7d88e48aa2bfb41136f5d0143b13dc88aa7e5a494c`
- Fixed-alpha sensitivity PDF:
  `a716d96e690b357e8516a2d904b8c24ebbc6bfd7be2303dfd803fcbdd66701ac`
- Cost-to-precision PDF:
  `1a9fa5860aa066f4cccac6c37b3bd28dd41037de893b59bb59b0145dd41815f2`
- Five-method plotting source:
  `eef0575af6ec1db1cfd63d216a2d8b0192e561d77b506c1dbefb3edef2008108`
- Fixed-alpha plotting source:
  `bdf5dab40ac6e5b7cbfedcb619b13f8d948bd28cdcc7de46039ddddd3241c73f`
