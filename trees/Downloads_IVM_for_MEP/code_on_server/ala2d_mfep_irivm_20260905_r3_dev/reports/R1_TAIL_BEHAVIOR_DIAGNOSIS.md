# Why the first IR-IVM attempt was fast early but worse in the noisy tail

Status: **diagnosis of the rejected r1 candidate; not formal evidence and not a
paper-figure replacement**.

The rejected r1 candidate used
\(\alpha=13\times10^{-5}\), \(r=0.5\), and \(m_{\max}=4\).  All comparisons
below use the same incremental number-of-mean-force-evaluations axis and the
same 20-image checkpoints.  The IR-IVM curve is the pointwise median of the
three completed repeats (seeds 41--43).

## Observed two-regime behaviour

For the early interval 20--300 mean-force evaluations (15 checkpoints), the
mean Riemannian normal residual was

| method | mean \(I_\perp^G\) |
|---|---:|
| IR-IVM r1 | 30.3967 |
| Current IVM | 49.9634 |
| FENEB (\(\lambda=0.02\)) | 130.0263 |

Thus the early acceleration is real: the rejected IR-IVM reached the correct
path neighbourhood much faster than both comparison methods.

For 1000--2000 mean-force evaluations (51 checkpoints), the ordering reverses:

| method | mean \(I_\perp^G\) |
|---|---:|
| Current IVM | 0.8951 |
| FENEB (\(\lambda=0.02\)) | 1.3377 |
| IR-IVM r1 | 1.6045 |

On the preregistered tail 1620--2000 (20 checkpoints), the normal-residual
median and mean are

| method | tail median \(I_\perp^G\) | tail mean \(I_\perp^G\) |
|---|---:|---:|
| Current IVM | 0.9445 | 0.9334 |
| FENEB (\(\lambda=0.02\)) | 1.2532 | 1.4107 |
| IR-IVM r1 | 1.4919 | 1.5857 |

The late disadvantage is therefore not a plotting illusion.  It is also not
an equal-arclength failure: the r1 article reparameterization residual stayed
near \(10^{-6}\), orders of magnitude below FENEB's displayed arclength
residual.  The tail free-energy-barrier errors were close--0.1308 kcal/mol for
Current IVM, 0.1288 kcal/mol for IR-IVM, and 0.1250 kcal/mol for FENEB--so the
specific failure is the average normal residual after the path has entered the
MD-noise-dominated neighbourhood.

## Algorithmic interpretation

The first IR-IVM attempt preserved its fast initial motion but kept converting
low-signal mean-force fluctuations into path motion.  Across the 1,200
image-state observations in the registered three-repeat tail, 48.3% of the
absolute image-wise normal mean forces were below one retained-frame-weighted
ten-block standard error and 76.4% were below two.  Internal images 15--20
contributed 78.4% of the aggregate tail
\(I_\perp^G\).  In that regime, choosing a new path from a noisy point estimate
can improve one checkpoint and worsen the next without improving the
underlying MFEP.

The V14 development design therefore retains the article reparameterization
and early frozen-frame step, but requires noise-aware residual bounds,
exact-centre resampling only when new samples can resolve an ambiguity,
frame-count-weighted block statistics, and a globally one-way curvature
takeover.  A candidate is not acceptable merely because its final point is
low.  It must retain the early advantage and beat Current IVM in the tail by
checkpoint win fraction, tail median, normalized tail area, and at least two
of three independent noisy repeats.  Since Current IVM's aggregate tail
\(I_\perp^G\) is already lower than FENEB's, passing that gate also removes the
specific late-NEB disadvantage seen here.

## Reproduction and provenance

The executable reproduction is
`analysis/summarize_r1_tail_behavior.py`; its machine-readable output is
`reports/R1_TAIL_BEHAVIOR_DIAGNOSIS.json`.  It first calls
`build_diagnostic_data()` in the sealed r1 diagnostic loader, selects the
stated NoF windows, and applies NumPy `mean` or `median` directly to the
returned `I_perp_G` and absolute barrier-error arrays.  For the image-wise
diagnosis it derives each block's retained-frame count from
`frame_counts - discarded_frame_counts`, verifies that their weighted mean
reconstructs the archived mean force, projects all blocks on the recorded
current Riemannian normal, and recomputes each image's contribution to
`I_perp_G`.  The loader validates the sealed registration, failed precheck,
all three completed public trajectories, archived Current-IVM and FENEB
inputs, and unchanged formal-PDF hashes before returning data.

Bound inputs:

| artifact | SHA256 |
|---|---|
| r1 campaign registration | `15c2608eb9268945a5bbc6a2364caa7ad25dbbc0cd98128a090ed91f0e3b9a7f` |
| r1 acceptance precheck | `ca13e61456196355c20e6986b0057ff7b37e1f831649b9313b80ea2219a5f7d2` |
| diagnostic data/plot loader | `bb043ecee263368744e35e837763ba92610f1bdde2f025e6b7ae2318ab0a618b` |
| archived Current-IVM trajectory | `b9cabc5119eaae01a2d1d6556b43219e7f34ae831f05bdbfec9a22787e145d24` |
| archived FENEB trajectory | `53f840618fcc194e1e3b288f9a8936020b4bac05990387068196601e2563fc49` |
| unchanged formal three-panel PDF | `ed57a7089730e760862ac435545d7e9fc8534601ee8c9008635b1a16d54b9932` |
| r2 diagnosis script | `ec4a809fc9887cb4edaad396e0659da57e83c329a76b1eb644a22c6300d5351b` |
| r2 machine-readable diagnosis | `39f2f2e13eabdc2392c6f08b37e61ae624cf4f413a6d31c4c86790fa1da3357e` |
