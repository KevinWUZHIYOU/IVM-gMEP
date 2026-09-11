# Ala2D V17 formal restrained-MD protocol — DRAFT

## Status and claim boundary

This document is a candidate-agnostic execution and acceptance draft.  It is
not a preregistered experiment and is not formal evidence.  The checked-in
registration has `status=DRAFT`, `formal_evidence=false`,
`new_restrained_md_authorized=false`, and `executable=false`.  It contains no
selected V17 parameter combination and no formal random seed.  Consequently,
the runner must fail before it acquires a lock, creates a run directory,
constructs an oracle, or invokes restrained MD.

In addition to those intentional placeholders, activation is technically
blocked.  The same V17 controller revision used as the r6 starting point failed
in the r5 development screen at seed 1701, trajectory ordinal 104, incremental
NoF 1940 with `an outer with completed movements must have anchor bounds`.
This r6 draft does not contain a fix or a regression for that real trajectory.
Its synthetic test suite therefore establishes draft-contract coverage only;
it does not establish that the controller is formal-ready.

The future formal claim is restricted to the artifacts under the future
write-once formal run root.  “No hidden mean-force evaluation” means that every
restrained-MD mean-force vector returned within that registered execution is
present in its atomic attempt records, append-only attempt ledger, and NoF
reconciliation.  It is not a claim about unrelated work elsewhere on the
machine.

## Activation order (future work, not authorized here)

The draft may not be edited in place to look retrospective.  After the separate
V17 development screen completes, create a new sealed registration in this
strict chronological order:

1. Diagnose the r5 NoF-1940 invariant failure in a new controller revision,
   add a reproducing regression, independently verify the repair, and only then
   complete and verify all 288 development trajectories.  Archive the complete
   analysis, execution manifest, plan, source hashes, and trajectory-ledger
   hashes.
2. Apply the already registered selection rule once and bind exactly one
   parameter combination by a write-once selection receipt.  Do not inspect any
   formal MD seed because none may yet exist.
3. Prospectively choose and attest three new optimization seeds, ten new paired
   terminal-diagnostic seeds, one new bootstrap seed, and one contour seed from
   the three optimization seeds.  They must be mutually compatible and must not
   reuse any development, pilot, or earlier formal seed.
4. Bind the exact V17 core, reparameterizer, geometry, oracle, analysis,
   acceptance, plotter, tests, protocol, MD inputs, common initial observation,
   current-IVM reference, FES, FENEB reference, GROMACS, PLUMED, Mutagen, and the
   unchanged formal PDFs by SHA-256.
5. Record the local hostname and one Mutagen synchronization session, including
   endpoint roots, version, mode, hash, connection/scanned state, and required
   idle status.  Seal a write-once execution manifest before the first formal
   mean-force evaluation.
6. Run the complete tests and have an independent reader verify that the formal
   run root is absent.  Only then may a separate, explicit activation set
   `new_restrained_md_authorized=true`, `formal_evidence=true`, and
   `executable=true` in the new registration.

Any change after activation requires a prospective addendum or a new campaign.
The r5 development seed outcomes may select the candidate, but they are never
relabelled as restrained-MD evidence.

## Algorithm and public states

The formal algorithm is the frozen V17 Riemannian inexact plus
reparameterized IVM core.  The normal residual, tangent, metric frame, and
article centered-tangent spring reparameterization use the same definitions as
the manuscript experiment.  `P_eff` can construct or diagnose a proposal, but
every own-merit, difference, lower bound, and upper bound uses only

\[
\Psi_{\mathrm{raw}}(\delta)
=-P_{\mathrm{raw}}^{\mathsf T}\delta
-\frac{\|\delta\|^2}{2\alpha}
\]

in the frozen source frame.

After takeover, a warm candidate whose source lower bound is nonpositive is
frozen byte-for-byte (path, normal displacement, frame, request, and hashes).
Exactly one independent exact-center band is then purchased at the source path.
Only its fresh raw blocks form the validation interval.  A positive lower bound
leaves the original frozen candidate pending for a later paid commit; a
nonpositive upper bound is confirmed bad; the crossing case is unresolved.
The latter two outcomes perform a bounded zero-NoF warm restart: the global
takeover latch remains set, H0 stays disabled, history is cleared, a new frame
is built, the scalar safeguard returns to `alpha`, and the curvature-epoch
floor prevents older pairs from reappearing.  One exact source path can restart
at most once.

The public ledger has one zero-cost common state followed by exactly three paid
state kinds:

- `reparameterized_frozen_inner_trial`: a purchased movement band;
- `exact_center_hold_resample`: a purchased repeated evaluation at the exact
  current path;
- `independent_exact_center_warm_validation`: the single purchased fresh
  validation band at the exact current path.

Both nonmovement kinds preserve the source path bytes and do not increment the
movement count.  A zero-NoF frame or warm-restart refresh is metadata on the
next paid state and never creates a new public cost point.

## Number of mean-force evaluations

One restrained-MD evaluation at one collective-variable restraint center that
jointly returns the complete two-component mean-force vector is one NoF.  The
two components do not multiply cost.  A complete 20-image band costs 20 NoF,
including a repeated-center HOLD or independent validation band.  Failed
attempts before a mean force is returned cost zero; a returned mean force costs
one even if a later eligibility gate rejects it, and that rejection makes the
formal run fail closed.

The common initial 20-image observation is shared with current IVM and is the
zero of the incremental axis; its absolute cost is 20.  Each of the three
formal optimization runs must publish exactly `0:20:2000`, hence 2000 new NoF
per run and 6000 in total.  Terminal diagnostics use three fixed anchors, ten
paired diagnostic seeds, three candidate paths plus one current-IVM path, and
20 centers per fixed path: `3 × 10 × 4 × 20 = 2400` NoF.  The planned total is
therefore 8400 new mean-force evaluations.  This number excludes failed
pre-mean-force attempts but their wall time remains reported.

No rejected proposal, recovery proposal, refresh, analysis, or plotting step
may query the MD oracle.

## Execution, crash recovery, and exclusivity

The formal runner uses `jobs=1` and one nonblocking host-global `fcntl` lock at
the workspace-level path `.ala_global_heavy_task.lock`.  The lock is shared by
all Ala campaigns on that host, not by only one seed directory.  A second Ala
job on the same host must fail without evaluating anything.  Cross-host formal
execution is not enabled by this draft; if it is later allowed, it additionally
requires a registered global claim/lease rather than relying on two independent
host locks.

After immutable registration and hash checks, the runner must acquire the lock,
repeat the registered Mutagen preflight, and publish the write-once execution
binding before it constructs the restrained-MD oracle.  The preflight fails if
the session, endpoints, version, mode, hash, connection/scanned state, or idle
status differs from the seal.

Every image attempt has one immutable logical identity and a unique attempt ID.
The atomic start/terminal JSON pair and the append-only JSONL ledger must match
one-to-one.  At most three byte-identical attempts may serve one logical
evaluation.  An existing validated result artifact is replayed; it is never
rerun or charged twice.  A public band is committed only after all 20 image
artifacts validate.  Replay reconstructs the controller from public states,
checks the checkpoint, and preserves all pre-crash and pre-resume bytes.

Reaching exactly 2000 NoF means that the requested evidence prefix is complete,
even if an independent validation at 2000 has proven a candidate whose movement
would cost another band.  Acceptance and plots use the actual source path in
the public validation state at 2000.  They do not buy the 2020 band, carry a
previous movement forward, or score the pending candidate.

## Acceptance in manuscript quantities

The three independent optimization paths are compared against the actual
current-IVM trajectory on the same incremental NoF axis.  `I_perp_G`, `I_arc`,
and the discretized mean-force barrier are computed with the manuscript
geometry and tangent.  Barrier quality is
`abs(Delta_F - Delta_F_star)`, where `Delta_F_star` is recomputed from the
sealed periodic FES minimax path before activation.

The frozen draft rules are:

- NoF 20–1600: the three-run median must have lower `I_perp_G`, no higher
  `I_arc`, and lower barrier error than current IVM at every checkpoint.
- NoF 1620–2000: noise-affected `I_perp_G` and barrier error must win at least
  80% of checkpoints, have lower tail median and normalized trapezoidal AUC,
  and at least two of three individual paths must have lower AUC.  The purely
  geometric `I_arc` median must be no higher at every checkpoint and at least
  two paths must have lower AUC.
- Every terminal path must pass endpoint, positive-definite metric,
  article-reparameterization residual, equal-arclength, segment jump, FES
  corridor, saddle proximity, and barrier-error guards.
- The paired terminal diagnostic treats each diagnostic seed as one bootstrap
  cluster across both noisy quantities, all three anchors, all candidate paths,
  and current IVM.  With 100000 fixed resamples, the one-sided 95% simultaneous
  upper bound on the maximum of the six mean paired log ratios must be strictly
  below zero; each noisy metric also requires lower AUC for at least two of the
  three candidate paths.

The acceptance producer and figure consumer both require
`ala2d.v17.formal_acceptance.v1`.  A missing exact state, incomplete attempt
ledger, early algorithmic stop, nonfinite metric, failed run, missing diagnostic
artifact, schema mismatch, or any hash mismatch is failure—not missing data to
be filled by last-observation-carried-forward.

## Formal figures

Formal rendering remains disabled until every acceptance section passes and an
explicit replacement flag is supplied.  The three-panel plot uses the median
and full range over the three prospectively registered optimization seeds.  The
four-panel contour uses the prospectively registered contour seed at exact NoF
0, 200, 400, and 2000; all three V17 paid states are valid exact anchors.  Both
plots label the horizontal axis “Number of mean-force evaluations” and the
method simply “IVM”.

Before replacement, render to a staging location, verify the PDFs visually,
recheck the pre-campaign formal PDF hashes, and emit a provenance JSON binding
the registration, acceptance report, public trajectories, baseline, FES,
FENEB, plotting source, and output hashes.  A failed acceptance leaves both
formal PDFs byte-for-byte untouched.
