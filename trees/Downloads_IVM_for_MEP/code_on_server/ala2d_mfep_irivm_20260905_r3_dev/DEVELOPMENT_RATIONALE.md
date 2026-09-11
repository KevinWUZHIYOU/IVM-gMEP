# Ala2D IR-IVM V15 development record

Status: **development only; not a preregistration and not formal evidence**.

This directory is a new, outcome-independent V15 development screen.  It does
not alter the earlier campaigns, authorize restrained MD, or replace either
formal paper PDF.  A later formal run still requires its own sealed
registration, source and input hashes, Mutagen preflight, host-local `fcntl`
lock, tests, and acceptance rule.

## Why V15 exists

The preceding frozen V14 screen completed all 48 planned trajectories and its
content-bound analysis returned `NO_COMBINATION_PASSED`; it recommended no
parameter combination.  The V14 plan and analysis hashes are, respectively,

```text
2399bfa55445374644d72faec09c29a6b645b31b9fd081faddbc90022a527dd8
fcbb44d0a90f0b4975de7196ff9664e6a5469d3f0fa447b25e82d7817e815c98
```

The failure exposed a controller problem under calibrated noise rather than a
reason to pick another isolated parameter.  V14 could turn two unresolved
questions into repeated exact-center HOLDs:

1. Its symmetric relative-residual interval required either a clear stop or a
   separate clear-continuation inequality.  Interval overlap requested another
   observation at the same path even though progress had not been ruled out.
2. An unresolved H0-versus-L-BFGS score difference could request HOLD even when
   H0 itself had a strictly positive improvement lower confidence bound.

This V14 outcome is used only as revision rationale.  No V14 path, metric,
diagnostic winner, or parameter ranking is an input to V15 optimization or
selection.  In particular, the V15 parameter grid, horizon, oracle seeds,
scientific acceptance rules, archived current-IVM baseline, and passing order
are copied unchanged.

## Frozen V15 change

V15 changes only the two decision branches above:

- The relative-residual test is one-sided and confidence-protected.  After at
  least one movement, the inner solve stops only when
  `U(P_j) <= r L(P_0)`.  If that condition is not established, including an
  overlapping interval, the controller continues an admissible movement in
  the same frozen outer until either a clear stop or `m_max`.  Residual
  ambiguity is diagnostic and is not by itself a reason to resample.
- Before the one-way warm takeover, L-BFGS is chosen only when its positive
  advantage over H0 is statistically resolved.  If that advantage is
  unresolved but H0 is valid and its own improvement lower confidence bound is
  strictly positive, the controller chooses H0.  If the required fallback H0
  has a nonpositive lower bound with positive reducible uncertainty, HOLD is
  allowed; a nonselected L-BFGS candidate cannot bypass the unresolved
  takeover comparison using its own lower bound.

The projected proximal equation, frozen Riemannian frame, article
reparameterization, pooled block statistics, reprojected secants, global
one-way warm takeover, bounded post-takeover scalar safeguard, public-state
publication, exact replay, and zero-query terminal semantics are unchanged.
After takeover, H0 remains forbidden.

## Fixed development experiment

Before any V15 result, the complete factorial is fixed as

```text
alpha in {10, 13, 16}e-5
r in {0.4, 0.7}
m_max in {4, 8}
```

These are the same 12 combinations as V14.  Each is run once with the
deterministic FES oracle and once on each calibrated synthetic-noise stream
1401, 1402, and 1403, for 48 exact trajectories.  Every trajectory starts from
the same causal initial path, has an incremental 0--2,000 development-query
horizon, and publishes only exact 20-query complete bands.  The integer alpha
mantissas and integer `m_max` values remain easy to state in the paper.

The algorithm cannot read `I_perp`, article arclength error, the barrier,
current IVM, FENEB, or the minimax reference path.  After completion, a fresh
deterministic surrogate oracle blindly scores every public path.  Actual
archived current-IVM paths, rescored by that same oracle, remain the sole
selection baseline; archived FENEB is contextual only.

The 20--1,600 pointwise rules; the 1,620--2,000 median, normalized-AUC,
win-fraction, and individual-seed rules; the deterministic guard; terminal
correct-path gates; and the lexicographic passing order are unchanged.  A
zero-query algorithmic terminal is a valid execution state, but it lacks the
exact 2,000 checkpoint and therefore cannot pass.  No last-observation-carried-
forward substitution is permitted.  If no combination passes, the analyzer
again recommends nothing.

## Cost language and execution boundary

One development-oracle estimate at one 2D collective-variable point jointly
returns both mean-force components and counts as one query equivalent.  A
complete 20-image band therefore costs 20 query equivalents.  These synthetic
and deterministic calls are not restrained-MD mean-force evaluations and are
not formal NoF evidence.

No screen is started by these design files.  The safe runner defaults to a
read-only `list`, writes a campaign manifest before any result if execution is
later requested, allows only one writer under a host-local lock, and binds the
V15 source, this frozen plan, specifications, inputs, runtime, and unchanged
formal PDFs.  The analyzer fails closed until all 48 exact 0--2,000
trajectories exist; `--allow-incomplete` can publish only a clearly labelled,
nonselecting inventory.

The complete numerical and state-machine contract is
`reports/V15_STATE_MACHINE_DESIGN.md`.  The development-only calibrated-noise
archive reuses the earlier retained mean-force blocks and adds zero MD NoF; its
provenance binds every contributing source.
