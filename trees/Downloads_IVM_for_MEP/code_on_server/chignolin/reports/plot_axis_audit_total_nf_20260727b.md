# Chignolin figure-axis audit — total Number of Forces

Date: 2026-07-27

## Sources inspected

- Latest manuscript: `/Users/wuzhiyou/Downloads/IVM_for_MEP/main.tex`
  - Chignolin section: lines 1811–2020
  - SHA-256: `db2068b461f07ff1f41f0c38e19999fa7f63cbda7ce61d748c96661c2b0faa4d`
- Generator for the figures currently cited by the manuscript:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/src/plot_fixed_alpha1_comparison.py`
  - SHA-256: `24b758440bf9329e8cce76ff2991949a8bb350891186831bccd9e533bc24775b`
- Revised five-method generator:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/src/plot_paper_method_comparison.py`
  - SHA-256: `80412c800836fc1ecbf664ead99a020f782f015a37782b78cdce3e5838e2d942`

## Finding and correction

The manuscript's cost-to-precision figure uses the correct orientation:

- horizontal axis: requested precision threshold;
- vertical axis: minimum first-passage cost needed to reach that threshold.

The first five-method version had these two quantities interchanged. The
revised generator now follows the manuscript convention while replacing the
old per-image/max-call cost by the total whole-band number of force
evaluations.

For the barrier panel, the displayed threshold interval is
`100 -> 19.5 kcal/mol`; this avoids using the several-thousand-kcal/mol
initial IDPP barrier as a visually uninformative precision target. Curves stop
at their actually attained precision and are never extrapolated.

## Axis contract used by the revised figures

| Figure type | Horizontal axis | Vertical axis |
|---|---|---|
| Raw convergence | Total `Number of Force Evaluations` | Physical metric |
| Diagnostics | Total `Number of Force Evaluations` | Diagnostic metric |
| Cost-to-precision | Precision threshold, decreasing left-to-right | Minimum total `Number of Force Evaluations` |
| Speedup | Precision threshold, decreasing left-to-right | ODE-NEB first-passage cost / method first-passage cost |
| Hyperparameter sensitivity | Hyperparameter | Metric at a stated total force-evaluation budget |
| Per-image counter audit | Image index | Method / cache statistic |

The optimization cost is computed as:

```text
counter_vectors.sum(axis=1)
```

Thus all 22 images are included, including the fixed endpoints' first
evaluation. Independent barrier-densification evaluations are excluded from
the optimization cost axis and are reported separately.

## Manuscript text that still uses the old accounting

The manuscript itself was inspected but not edited in this audit. It still
states the old maximum-per-image convention at lines 1902–1904 and uses
per-image call wording/numbers in the captions and discussion at lines
1944–1946 and 1965–1987. Those passages must be updated when the revised
five-method figures are inserted.

## Corrected output

Directory:

`/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/figures/paper_methods_total_nf_20260727b`

Corrected cost-to-precision PDF:

`chignolin_cost_to_precision_total_nf.pdf`

SHA-256:

`8ec9c2bed9b1486d374b67d62aeabbc3864caa95b4923499e45d96b800997e48`

The PDF is one page, has embedded fonts, and the PDF/SVG/600-dpi PNG variants
were regenerated from the same source.

## Validation

- All generated SVG files parse successfully.
- The corrected cost-to-precision PDF renders without clipping or overlapping
  panel labels.
- The project-prescribed test command passes all 53 tests.
