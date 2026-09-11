# Ala `r=0.1` with-reparameterization inner-residual figure

Status: frozen-input validation, formal figure generation, and visual
inspection completed on 2026-09-09.

## Scientific quantity

The prescribed inner stopping parameter is fixed at \(r=0.1\).  Therefore the
vertical coordinate is not a varying value of \(r\), but the actually attained
global natural-residual ratio at each fixed-path solve endpoint,
\(R_k^{\mathrm{end}}/R_k^0\).  The horizontal reference line is the prescribed
threshold \(r=0.1\).

Only solve-endpoint residuals are available in the frozen optimizer artifacts.
No inner-step residual trace is reconstructed.  Reparameterization and recovery
force calls remain included in cumulative Number of Forces; they appear as
horizontal gaps between successive solve endpoints and receive no invented
residual values.

## Frozen experiment inputs

- optimizer preregistration:
  `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_campaign_v1_1788923892307558000_preregistered.json`
  (`da1ded53f6dc11ac71ebe5bae64ba452f19c95fdc60467be220ff630912b6a05`)
- local report:
  `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_campaign_v1_1788923892307558000_local_report.json`
  (`78ef8b82b7b7a8dbc2b6f4f62d8728fb216c1494fce46a95bbddce5241592362`)
- aggregate:
  `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_inexact_reparam_alpha_r01_c5000_campaign_v1_1788923892307558000_aggregate.json`
  (`50914e1cdfdafc59287921b4eef6081a5274d40366a4da926a590c06a6c9828a`)

Every optimizer-result SHA-256 is frozen inside the plotting wrapper and is
rechecked before any output is written.

## Exact endpoint alignment

`discrete_audit_global_relative_residual` and
`discrete_audit_stop_reason` contain one item per completed fixed-path solve.
`discrete_ffrbk_selector_audit_solver_call` and `discrete_audit_step` identify
the paid inner steps.  From the cost-aligned history the plotting code selects
`reparam_attempted == false`, skips the initial \(N_F=20\) record, consumes the
registered number of inner steps for each solve, and pairs the last such record
with the solve-level residual.  The corresponding
`total_internal_joint_evaluations` value is the exact cumulative force cost.

The validator permits at most one additional non-reparameterization history
record, and only as the final ledger closure after a budget interruption.  It
also verifies contiguous solve IDs, inner-step numbering, strict NoF growth,
the 5000-NoF terminal ledger, stop reasons, the nonbinding implementation cap,
and equality between each saved NoF and the sum of the 20 internal-image
counters.

## Plotting code and outputs

- plotting wrapper:
  `paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_r01_reparam_inner_residual_vs_nf.py`
  (`769a33ee28880455edee9f1d94dd2983d32c951599e0850a72536b66b526a1f9`)
- PDF:
  `output/pdf/alanine66d_r01_reparam_inner_residual_vs_nf.pdf`
  (`be7a280fb970b5fd0bde214e5c5ee9d8d32a5017c9e05c7c588f802519a03811`)
- source table:
  `output/pdf/alanine66d_r01_reparam_inner_residual_vs_nf_source.csv`
  (`1bd326fe887fce365385a71f3749512cf9391fc6c9a9fca43965dd78a90b1fad`)
- machine-readable manifest:
  `output/pdf/alanine66d_r01_reparam_inner_residual_vs_nf_manifest.json`
  (`728bda0f37e825a9bf73f02bdd94c15cdbaf4de15abfc0a39ef1fd6834207d1a`)

The final figure uses the same eight discrete `jet` colors and marker sequence
as the registered 2x3 alpha comparison.  Residual-stop endpoints are filled
markers; a final per-image-call-limit stop is a hollow `X`.  Both axes are
logarithmic.  The source table records only observations; interpolation,
extrapolation, and carry-forward are all disabled.

## Formal generation and validation

The formal command was:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig_residual_qa \
  /Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_r01_reparam_inner_residual_vs_nf.py
```

The wrapper revalidated all three frozen campaign reports and all eight frozen
optimizer-result hashes before writing outputs.  The CSV contains 245 data
rows, exactly matching the manifest total.  Counts by alpha are 155, 58, 16,
4, 3, 3, 3, and 3 for
\(\alpha=10^{-1},10^0,\ldots,10^6\), respectively.  Four final solves,
for \(\alpha=10^3,10^4,10^5,10^6\), stop at the per-image call limit and
do not satisfy \(r=0.1\); all other 241 endpoints satisfy the prescribed
residual threshold.  No solve stops at the \(m_{\max}=5000\) implementation
cap.

`pdfinfo` reports one unencrypted PDF 1.4 page of size
504.567 by 289.134 points.  `pdffonts` reports the Times New Roman and STIX
TrueType subsets as embedded.  The PDF was rendered with Poppler at 200 dpi
and inspected at native resolution.  Labels, logarithmic ticks, markers,
legend, threshold line, and the four hollow call-limit markers are legible;
there is no clipping, overlap, missing glyph, or rasterization defect.
