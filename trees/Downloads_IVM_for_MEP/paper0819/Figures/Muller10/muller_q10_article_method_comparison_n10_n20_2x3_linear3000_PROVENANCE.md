# Q-transformed Muller comparison with old and new IR-IVM

This record traces the redraw of `muller_q10_article_method_comparison_n10_n20_2x3_linear3000.pdf` in the `paper0819` snapshot.

## What changed

- The first two columns now include `Previous IR-IVM` and `IR-IVM` for both 10 and 20 interior images.
- The six article methods and all of their plotted values are unchanged from the registered source figure.
- The third-column path contours are unchanged and retain the original IVM, ODE-NEB, and OBSM-LBFGS checkpoint policy.
- The horizontal axis remains total number of force evaluations, 0--3000, on a linear scale.
- The second metric remains the raw 500-point coordinate-wise linearly densified energy barrier, not its error relative to a reference value.

## Meaning of the two IR-IVM curves

- `Previous IR-IVM` is the exact-envelope predecessor of the final unified run.
- `IR-IVM` is the final horizon-first unified run.
- For 10 interior images, the two trajectories are bitwise identical.
- For 20 interior images, they differ only at 80 force evaluations and are bitwise identical from 100 force evaluations onward. The final run has no recorded joint-metric rise; the predecessor has one.

## Reproduction files

- Plot source: `plot_muller_q10_article_method_comparison_n10_n20_2x3_with_irivm.py`
- Frozen IR-IVM inputs: `data/irivm_overlay/`
- Exported plotted values: `muller_q10_article_method_comparison_n10_n20_2x3_linear3000_source.csv`
- Machine-readable provenance and hashes: `muller_q10_article_method_comparison_n10_n20_2x3_linear3000_manifest.json`

The registered base PDF had SHA-256 `9f2350844a42311a12bac182fd3dc5e13f83d5b775040fae028d6ede2a2f9be7`. The redrawn original-method rows were compared with the registered source CSV: all 3,766 rows were retained, with maximum absolute differences of zero for both plotted metrics. The redraw adds 352 IR-IVM rows.
