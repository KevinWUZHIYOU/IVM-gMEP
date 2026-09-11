# Ala2D V17 formal campaign draft (r6)

This directory is a candidate-agnostic implementation draft.  It is not a
formal experiment registration and it grants no restrained-MD authority.

**Known blocking defect (2026-09-05):** the running r5 development screen
stopped at seed 1701, trajectory ordinal 104, incremental NoF 1940 with the
invariant failure `an outer with completed movements must have anchor bounds`.
r6 currently carries the same V17 controller revision (apart from formal-draft
identity/loader scaffolding), so this defect is not fixed or regression-tested
here.  Passing synthetic tests does not make this core formal-ready.  A new
controller revision, a reproducing regression test, and an independently
verified development rerun are required before any r6 activation work.

The only registration currently present is
`registrations/V17_FORMAL_CAMPAIGN_DRAFT.json`.  Its status is `DRAFT`, its
evidence and authorization flags are false, and all parameter and seed fields
are explicit placeholders.  Every execution entry point must therefore fail
closed before creating a result directory or calling an oracle.

After the controller defect is fixed in a new revision and the separate
development screen has produced and analyzed all 288 registered trajectories,
a new immutable registration may select one parameter triple and prospectively
register new optimization, diagnostic, bootstrap, and contour seeds.  That
later registration must be independently sealed and is not created by this
draft.

No `runs/` directory, MD input archive, development output, or formal PDF is
copied into r6.  The sole 9 KiB NPZ under `tests/fixtures/` is explicitly
test-only, lets the zero-MD V17 state-machine tests remain self-contained, and
cannot satisfy any future formal input gate.

The full non-executable protocol is in `PROTOCOL_DRAFT.md`; the chronological
activation checklist and file inventory are in
`reports/V17_FORMAL_ACTIVATION_PLAN.md`.
