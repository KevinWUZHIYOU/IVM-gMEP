# V17 formal MD draft: execution, evidence, and plotting contract

## Current status

This document describes implementation scaffolding, not an experiment that may
run.  The companion registration has status `DRAFT`, `formal_evidence=false`,
`new_restrained_md_authorized=false`, and `executable=false`.  The selected
V17 controls and every new seed remain named placeholders until the complete
r5 288-trajectory screen is analyzed.  r6 contains no `runs/` tree and no MD
input archive.

The inherited V17 controller is also not formal-ready.  r5 stopped at seed
1701, trajectory ordinal 104, incremental NoF 1940 on the invariant `an outer
with completed movements must have anchor bounds`.  r6 has not repaired or
reproduced that failure.  The tests below are useful contract tests, but their
success cannot override this observed development failure.  Activation is
blocked pending a new core revision, a targeted replay/regression, and an
independently verified development rerun.

## Algorithm identity and public states

The formal core is the V17 raw-consistent frozen-frame algorithm.  Proposal
directions may use the effective residual, but every own-merit, comparison,
validation, and movement interval uses the frozen raw proximal quantity.  A
post-takeover candidate with nonpositive source lower evidence receives exactly
one independent exact-center validation band; a confirmed bad or unresolved
candidate receives the bounded zero-cost warm restart, while a validated
positive candidate is committed only by a later separately paid band.

State zero is `common_initial`.  Every later public cost point has exactly one
of three kinds:

1. `reparameterized_frozen_inner_trial`: a paid movement.
2. `exact_center_hold_resample`: a paid source-path HOLD.
3. `independent_exact_center_warm_validation`: a paid, fresh source-path
   validation that does not move.

All three paid kinds cost twenty NoF because each evaluates twenty internal CV
centers.  Repeating a center costs again.  A zero-NoF outer refresh or bounded
warm restart creates no additional public point and is deterministically
rederived after a crash.

If the exact 2000-NoF public state is an independent validation with a positive
candidate pending, acceptance and plotting use the actual source path stored at
2000.  They neither carry forward an older movement nor inspect the unpaid
candidate nor purchase a 2020 band.

## NoF and artifact audit

One restraint center returning the complete two-component mean-force vector is
one NoF, not two.  A scoreable optimization trajectory must have exactly 101
public states at `0,20,...,2000`.  Its 100 paid bands must match exactly 2000
unique logical `(band_id,image_index)` evaluations.  Attempt retries retain
distinct attempt IDs, but at most one complete finite result artifact may
supply each logical evaluation.  Failed-before-mean-force attempts cost zero;
a returned complete vector costs one even if a separate eligibility gate then
fails.  No candidate, recovery, diagnostic, or cache evaluation may exist
outside the registered ledger.

## Concurrency and recovery

Formal execution is serial (`jobs=1`) and must hold a host-global fcntl lock
shared by every Ala campaign on that host.  A campaign-local lock is
insufficient.  Registration binding, atomic per-attempt records, public states,
the replaceable controller checkpoint, and a public-state hash ledger jointly
support exact resume.  A public state written immediately before a crash is
replayed into its controller checkpoint without another oracle call.  A
completed image artifact is reconciled, never repeated or double charged.

## Acceptance and figures

Optimization acceptance retains the registered article-language comparisons:
the three-run median must beat current IVM throughout the front/middle region,
the tail must satisfy checkpoint, median, AUC, and individual-run conditions,
and all trajectories must pass the correct-path guard.  Only after that
precheck may paired fixed-path terminal diagnostics run.  The diagnostic seed
is the bootstrap cluster, preserving both noisy quantities and all three
anchors jointly.

Acceptance production and the plot-data builder share the single schema
`ala2d.v17.formal_acceptance.v1`.  This removes the prior producer-v2 versus
consumer-v1 mismatch.  A draft can validate fixtures and build no formal PDF;
formal replacement additionally requires an explicit command, passed final
acceptance, unchanged target hashes, and the later sealed registration.

## Legacy-contract repairs

The top-level implementation, public states, acceptance producer, and plotter
are bound to the V17 identity.  The old active optimization seeds 41–43 appear
only in the explicit forbidden-prior-seed list and in a negative test; they are
not executable defaults.  The acceptance and contour selectors read the future
prospective seeds from the sealed registration.  No consumer assumes that every
paid row is a movement: movement, exact-center HOLD, and independent validation
are all charged and scoreable, while their action flags and path-byte semantics
are checked separately.

The unchanged article spring implementation retains its own historical
subcomponent audit identifier.  That identifier is not used as the IR-IVM
algorithm identity and does not cause the formal runner or plotter to claim the
old V13 controller.

## Draft execution boundary

Only `validate-draft` can succeed against the checked-in registration.  The
four mutation-capable commands validate the non-executable registration before
hash probing, directory creation, lock acquisition, oracle construction, or
external execution, then fail closed.  A future activation is required to add
the candidate selection receipt, entirely new seed attestation, source/input/
runtime hash tables, named local host, Mutagen session seal, and formal run
adapter.  Merely changing the status or copying the core cannot create valid
formal evidence.

The future seal fixes `jobs=1`, one workspace-level host-global `fcntl` lock,
three attempts per byte-identical logical evaluation, and mandatory Mutagen
preflight.  Cross-host execution is false in this draft; enabling it would also
require a registered global claim/lease.

## Verification surface

The unit tests use only analytic or synthetic oracles.  They cover V17 raw
score behavior, all validation branches, the bounded warm restart and epoch
floor, exact public-state replay, crash/resume prefix invariance, terminal
pending commit, all three paid state kinds, absolute and incremental NoF,
fresh-only validation evidence, zero-NoF refresh metadata, atomic/JSONL attempt
uniqueness, retry identity and limit, hidden evaluation rejection, draft
side-effect ordering, jobs=1, cross-process lock exclusion, acceptance schema,
and plotter fail-closed behavior.  Tests create no campaign run tree and spend
no restrained-MD NoF.

They do not include the r5 seed-1701/ordinal-104/NoF-1940 artifact prefix,
which is intentionally not copied into this candidate-agnostic draft.  That
missing real-trajectory regression is a known activation blocker rather than
evidence that the failure is absent.
