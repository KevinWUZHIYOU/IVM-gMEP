# R6 V17 formal-draft test contract

Status: independent design review for a non-executable draft.

Known activation blocker: the inherited V17 controller failed during the r5
development screen at seed 1701, trajectory ordinal 104, incremental NoF 1940
with `an outer with completed movements must have anchor bounds`.  This r6
draft has neither fixed that fault nor imported the real failing prefix as a
regression.  The contract below remains useful, but even a fully green
synthetic suite is not evidence that this controller revision is formal-ready.

This document defines the minimum safe API and the tests that must pass before
the Ala2D V17 formal campaign can be sealed.  It authorizes no restrained MD,
does not choose the final controls or seeds, and does not make development
screen output formal evidence.  Until the top-level registration field is
exactly

```json
{"status": "SEALED_BEFORE_FIRST_FORMAL_MD"}
```

all execution entry points must fail closed.  The current draft value is
exactly `"DRAFT"`.

## 1. Vocabulary and non-negotiable invariants

There are four public state labels.  State 0 is historical shared evidence and
is not a newly paid state.  Every later state is one complete, newly paid band.

| public `state_kind` | role | incremental band cost | movement | public path |
|---|---|---:|---:|---|
| `common_initial` | shared initial state, index 0 | 0 | 0 | shared initial path |
| `reparameterized_frozen_inner_trial` | paid movement | 20 | 1 | evaluated candidate path |
| `exact_center_hold_resample` | ordinary paid HOLD | 20 | 0 | exact source/current path |
| `independent_exact_center_warm_validation` | paid independent validation | 20 | 0 | exact source/current path |

The two no-movement kinds are not interchangeable.  An ordinary HOLD pools a
new exact-center band for the ordinary anchor/pre-takeover policy.  Independent
validation evaluates one already frozen post-takeover candidate using a fresh
band at its source.  It must retain the exact V17 validation fields and hashes.

For state index `i`:

```text
nof_mean_force_incremental == 20 * i
nof_mean_force_absolute_including_shared_initial == 20 * (i + 1)
```

Thus a complete 0:20:2000 trajectory has exactly 101 public states.  NoF means
the number of mean-force-vector evaluations at individual collective-variable
points.  The two components returned at one point jointly count as one NoF;
the 20 internal images make one paid band cost exactly 20 NoF.

At every ordinary HOLD and validation row, all of the following are required:

- `full_path_unwrapped` is byte-identical to `source_paid_path_unwrapped` and
  to the immediately preceding current path;
- `paid_action_is_movement == false` and the movement counter is unchanged;
- the band is nevertheless new, complete, public, and costs exactly 20 NoF;
- reparameterization/candidate construction is not charged as NoF;
- no future frozen-candidate path is copied backward into this public row.

At a paid movement row, the actual evaluated and published path is the exact
reparameterized frozen candidate.  Its movement counter increases by one.  A
structurally valid V17 movement has a nonzero actual normal displacement, so a
movement row cannot masquerade as a byte-identical HOLD.

## 2. Minimum safe API

The implementation need not introduce a large framework.  It does need one
shared public-trajectory contract, one sealed-registration gate, and one paid
oracle boundary.

### 2.1 One shared public-trajectory contract

Create one small module, imported by the core loader, acceptance code, and
plotter, that owns these constants and no duplicated aliases:

```python
COMMON_INITIAL_STATE_KIND = "common_initial"
MOVEMENT_STATE_KIND = "reparameterized_frozen_inner_trial"
HOLD_STATE_KIND = "exact_center_hold_resample"
VALIDATION_STATE_KIND = "independent_exact_center_warm_validation"
PAID_STATE_KINDS = frozenset({
    MOVEMENT_STATE_KIND, HOLD_STATE_KIND, VALIDATION_STATE_KIND,
})
N_INTERNAL = 20
BAND_NOF = 20
REGISTERED_BUDGET_NOF = 2000
```

The same module should expose only the following consumer-facing operations
(names may differ, semantics may not):

```python
validate_public_trajectory(trajectory, *, expected_implementation_id,
                           required_budget=None) -> ValidatedTrajectory
exact_public_state_index(validated, nof: int) -> int
terminal_public_state_index(validated, budget: int) -> int
```

`validate_public_trajectory` checks identity, array alignment, exact state
labels, shapes, finite required metrics, contiguous indices, exact 20-NoF
increments, path/action invariants, and pending/completion consistency.  The
two selectors require an exact public checkpoint.  They never use
last-observation carry-forward and never search backward for the latest
movement.

The validated view needs, at minimum, aligned read-only arrays for
`state_index`, `nof_mean_force_incremental`, `state_kind`,
`full_path_unwrapped`, `M`, `grad_F`, `I_perp_G`, `I_arc`,
`Delta_F_barrier`, `segment_length_relative_deviation`, and the V17
reparameterization residual.  It also carries the exact implementation ID,
run-manifest digest, completion/pending status, and a digest of the immutable
public-state sequence.  Acceptance and plotting must consume this validated
view (or a serialized object produced by it), not an unchecked dictionary.

Legacy labels such as `reparameterized_inner_trial`, a single
`PUBLIC_STATE_KIND`, or a V13 implementation ID are inadmissible in the R6
acceptance and plotting path.

### 2.2 Registration and execution gate

Use one registration loader with two explicit modes:

```python
load_registration(path, *, require_execution_ready: bool)
assert_formal_execution_ready(registration, exact_bytes, campaign_root)
```

Read-only draft inspection may call the first function with
`require_execution_ready=False`.  Every operation that could construct a
formal oracle, acquire an execution lock, create a run/band/attempt path,
resume a run, launch a subprocess, or publish a formal figure must call it with
`True` first.

Execution readiness requires, in order:

1. exact `status == "SEALED_BEFORE_FIRST_FORMAL_MD"`;
2. no missing, `null`, empty, `TBD`, `PLACEHOLDER`, or `UNRESOLVED` value in
   any execution-critical field;
3. one final V17 formal implementation ID and algorithm variant, identically
   bound by registration, source, run manifest, checkpoint, public rows,
   acceptance report, and plotter;
4. final controls, parameter ID, all seed roles, NoF schedule, run order,
   inputs, executables, source hashes, host, Mutagen session, and lock path;
5. a registration digest sealed before the first atomic attempt record.

Merely changing `DRAFT` to the sealed string must not make a registration with
placeholders executable.

### 2.3 One paid oracle boundary

Candidate generation, article reparameterization, frozen-frame construction,
merit calculation, validation classification, metric calculation, zero-NoF
refresh/restart, serialization, replay, acceptance, and plotting are pure with
respect to mean-force evaluation.  Only the run loop may call
`oracle.evaluate_band`, at one auditable paid-transition boundary.

A prepared paid request contains at least the next state/band index, exact
path bytes and path digest, one of the three paid `state_kind` values,
parameter ID, master seed, registration digest, and source-checkpoint digest.
The returned band is accepted only when it has all 20 distinct image indices,
the exact requested centers, a complete finite two-component mean-force
vector and eligible metric per image, all block data, deterministic seeds,
runtime provenance, and `nof == 20`.

No convenience API may evaluate a candidate, validation diagnostic, metric,
acceptance gate, or plot outside this paid boundary.

### 2.4 One host-global Ala lock

All formal execution stages use one nonblocking host-local `fcntl` lock with
`jobs=1`.  “Host-global Ala” means its identity is independent of checkout,
campaign revision, parameter, seed, stage, and run directory.  A per-run or
`r6`-suffixed lock is insufficient.  Because r1 is not to be modified, the r6
registration must point at a canonical Ala lock path that also excludes any
contemporaneous legacy Ala launcher on this host.

The lock is acquired after the read-only sealed-registration/hash/host checks
but before the first mutable campaign action or oracle construction, and is
held for the entire stage/resume command.  Failure to acquire it creates no
run artifact and starts no subprocess.  The lock file must be a regular,
non-symlink path with mode 0600; owner metadata are diagnostic and do not
replace `fcntl` ownership.

## 3. Mandatory test matrix

All tests below use a counting fake oracle and temporary directories.  They
must not launch GROMACS, PLUMED, or any real restrained MD.

### A. Draft and registration gates

| ID | fixture/action | required result |
|---|---|---|
| A01 | top-level `status: "DRAFT"`; call every run/resume/confirm/diagnostic command | each fails before lock, directories, oracle construction, subprocess, or attempt record |
| A02 | draft; request formal rendering | fails before output creation or replacement |
| A03 | missing, lower-case, Boolean, or unknown status | fail closed; no permissive truthiness or case folding |
| A04 | sealed status but final parameter is `null`/sentinel | fail before side effects |
| A05 | sealed status but any seed role is unresolved | fail before side effects |
| A06 | sealed status but source/input/runtime/hash/host/Mutagen/lock binding is absent or changed | fail before oracle call; report the first exact mismatch |
| A07 | draft read-only validation/listing | may return a clearly marked non-executable inventory, with zero writes and zero oracle calls |
| A08 | import every module and run CLI `--help` | zero filesystem mutation and zero oracle/subprocess calls |
| A09 | attempt a CLI override of controls, seed, budget, `jobs`, lock path, host, or registration hash | reject rather than override registration |
| A10 | change only the status of a placeholder draft to the sealed value | still reject all unresolved execution-critical fields |

The gate test should monkeypatch the lock constructor, oracle constructor,
`subprocess.run`, and every atomic writer to raise if reached.  This proves the
gate order, rather than merely checking an error message.

### B. V17 identity, controls, and seeds

| ID | mutation | required result |
|---|---|---|
| B01 | one valid V17 formal identity across all bound artifacts | accepted by the shared loader |
| B02 | substitute the r5 development identity | reject as non-formal evidence |
| B03 | substitute any V13 identity or state label | reject, even if numerical arrays are valid |
| B04 | mix identities among registration, run manifest, checkpoint, rows, acceptance report, and plot input | reject at the first boundary |
| B05 | source bytes differ from the registered V17 source digest | reject execution and consumption |
| B06 | controls differ from registered `(alpha, r, m_max)` or directory/run binding | reject; no “close enough” float comparison |
| B07 | invalid `alpha`, `r`, or `m_max`; include NaN, infinity, Boolean-as-integer, `r >= 0.8`, and `m_max <= 3` | reject before a run is created |
| B08 | duplicate seed, Boolean seed, seed-role overlap forbidden by registration, or seed mismatch in run/attempt | reject |
| B09 | parameter/seed values are final but `parameter_id` is stale | reject; ID and exact values are one binding |
| B10 | final seed/parameter values supplied after any formal attempt record exists | sealing must refuse |

The formal-draft V17 implementation ID identifies this scaffold, but does not
claim that its inherited controller is ready to seal.  The selected parameter
and formal seeds are deliberately not chosen in this document.  In the draft
JSON they must remain unmistakable placeholders, never reasonable-looking
numbers such as 0, -1, or a development seed.

### C. Public state schema and transition truth table

| ID | fixture/mutation | required result |
|---|---|---|
| C01 | valid index-0 row | kind is exactly `common_initial`, incremental NoF 0, absolute NoF 20, no new attempt, not paid |
| C02 | valid movement row | exact movement kind, one band/20 NoF, movement +1, evaluated path is frozen reparameterized candidate |
| C03 | valid ordinary HOLD | exact HOLD kind, one band/20 NoF, movement +0, exact source path/certificate reuse, not validation |
| C04 | valid independent validation | exact validation kind, one band/20 NoF, movement +0, exact source path, fresh validation record |
| C05 | unknown or legacy state kind | shared loader rejects |
| C06 | kind/flags disagree (for example validation kind with validation false, HOLD kind with movement true) | reject |
| C07 | HOLD/validation changes one path bit, endpoint, state index, or movement counter | reject |
| C08 | movement row reuses exact source bytes or lacks its reparameterization certificate | reject |
| C09 | noncontiguous state indices, nonmonotone NoF, a 10/19/21-NoF step, duplicate row, or missing row | reject |
| C10 | aligned field has wrong length/shape/dtype, object dtype, missing key, or nonfinite required value | reject |
| C11 | immutable public path already exists with different bytes | publish-once collision; never replace |
| C12 | alter a public row after checkpoint/replay | digest/replay failure before analysis or plot |

Use one valid fixture for each state kind, then mutation-test every invariant
one at a time.  This is more likely to catch cross-field drift than tests that
construct only the normal path.

### D. Independent validation and pending movement

| ID | fixture/action | required result |
|---|---|---|
| D01 | post-takeover structural candidate with source `L_raw <= 0` | freeze exact source, candidate, actual displacement, frame, target, full construction, and hashes before any validation query |
| D02 | issue validation request | requested/evaluated path is byte-identical source; new band ID and derived seeds make it independent despite identical centers |
| D03 | fresh `L_raw > 0` | classification `validated_positive`; equality at zero is not positive |
| D04 | fresh `U_raw <= 0` | classification `confirmed_bad`; equality at zero is bad |
| D05 | `L_raw <= 0 < U_raw` | classification `unresolved` |
| D06 | craft source/pooled blocks and fresh blocks to imply opposite classifications | branch follows fresh raw blocks only; source pooling and `P_eff` contamination are detected |
| D07 | validation publication | +20 NoF, zero movement, validation state kind, exact source path, fresh evidence and frozen hashes public before later action |
| D08 | positive validation without pooled r-stop and budget for another band | checkpoint pending exact frozen candidate; next PREPARE bypasses adaptation and evaluates precisely that candidate for another +20 and +1 movement |
| D09 | positive validation with pooled r-stop | pooled r-stop has precedence; frozen candidate discarded without movement |
| D10 | bad/unresolved without pooled r-stop | one zero-NoF same-path warm restart; no second independent validation at that path |
| D11 | candidate structurally invalid | typed zero-NoF terminal; no validation band and no oracle call |
| D12 | pre-takeover ambiguity | follows registered H0/HOLD policy; cannot use post-takeover independent validation |

The D06 fixture is essential.  It should make a pooled/source calculation
positive while the fresh calculation is bad, and a second case should reverse
the signs.  Merely asserting a Boolean “fresh only” field does not test the
mathematics.

### E. Exact 2000-NoF pending-source horizon

Construct a complete 101-row trajectory whose row at NoF 2000 is a positive
`independent_exact_center_warm_validation` and whose frozen candidate would
need a separately budgeted band at NoF 2020.

| ID | assertion | required result |
|---|---|---|
| E01 | completion | `completed`, `budget_prefix_complete`, `requested_prefix_complete`, and `budget_fully_spent` are true at 2000 |
| E02 | pending fields | pending action and pending validated commit are true; status/phase identify the frozen pending commit |
| E03 | terminal path | exact row-2000 path equals row-1980 source/current path; it is not the frozen future candidate |
| E04 | terminal correct-path gate | acceptance scores the actual row-2000 source/current path and its row-2000 estimates |
| E05 | metric curves | acceptance uses the exact row-2000 metric values; no LOCF from 1980 and no fabricated 2020 value |
| E06 | contour snapshot at 2000 | plotter draws the actual row-2000 source/current path |
| E07 | resume with unchanged 2000 budget | zero oracle calls and byte-identical completion/pending artifacts |
| E08 | resume under a separately registered 2020 budget fixture | exactly one new 20-NoF candidate band; no duplicate validation; candidate digest equals frozen pending digest |
| E09 | remove or mutate pending hashes/phase/flags while leaving the path plausible | core loader, acceptance, and plotter all reject |
| E10 | replace the validation kind with movement to satisfy a legacy selector | reject kind/flag/hash mismatch |

This case resolves the consumer rule: every exact paid public state is
scoreable.  “Latest reparameterized movement at or before the budget” is not a
valid selector for V17.  The ordinary HOLD and independent-validation paths
are already article-reparameterized source paths; spending a band without a
movement is part of the algorithm's cost-to-precision trajectory.

### F. Zero-NoF refresh/restart

| ID | fixture/action | required result |
|---|---|---|
| F01 | trigger registered trust refresh | same state index, same NoF, no new public state, no attempt record, no oracle call |
| F02 | compare protected controller fields before/after | path, paid `M`/gradient/blocks, observations, movement count, and takeover latch unchanged; only registered frame/history/marker fields change |
| F03 | next paid transition after refresh | exactly one new public paid row records the refresh event and costs 20 |
| F04 | crash after refresh preparation but before oracle | refresh is rederived; no same-index checkpoint/public row and no extra evaluation |
| F05 | second refresh at the same exact path or refresh in an already fresh frame | durable typed zero-NoF terminal |
| F06 | bad/unresolved validation warm restart | zero NoF and no public row; marker and curvature epoch persist until genuine movement |
| F07 | resume a durable zero-NoF terminal | deterministic rederivation, no oracle call, no state/attempt append |
| F08 | zero-NoF event code tries to create a state, advance NoF, or append attempt ledger | injected writer/counter raises and test fails |

### G. No hidden mean-force evaluation and attempt-ledger uniqueness

| ID | fixture/action | required result |
|---|---|---|
| G01 | call every pure preparation/reparameterization/metric/replay/acceptance/plot function with counting oracle available | oracle count remains zero |
| G02 | complete each of the three paid state kinds | exactly one band request and 20 returned mean-force-vector evaluations |
| G03 | return `OracleBand.nof != 20`, missing image, duplicate image, wrong center, or incomplete vector | reject; never publish a paid row |
| G04 | successful band audit | image indices are exactly 0..19; each has one eligible NoF-returning result bound to the state/band index |
| G05 | duplicate attempt ID or two terminal records for one attempt | reject |
| G06 | two NoF-returning attempts for one logical `(band,image)` | reject; never silently choose one |
| G07 | retry after `failed_before_mean_force` | same logical identity/seeds/center, new unique attempt ID, previous attempt NoF 0, maximum registered attempts enforced |
| G08 | `mean_force_returned_but_ineligible` | charge/report one realized NoF, fail the band, do not retry or make it scoreable |
| G09 | valid result artifact exists but terminal record was interrupted | reconcile as one `recovered_success`; do not rerun or double-charge |
| G10 | terminal success exists but artifact is missing/corrupt, or identity/hash differs | fail closed; no retry pretending the first NoF did not happen |
| G11 | re-open a completed band | byte-identical reuse, no new attempt records, no new JSONL lines, no new NoF |
| G12 | atomic records versus JSONL | exact bijection after reconciliation; no duplicate identical JSONL append on replay |
| G13 | completed trajectory audit | every paid public state joins to exactly one complete band and 20 eligible image results; no unjoined complete band |
| G14 | static call-site audit | the formal source has one authorized `evaluate_band` call site; acceptance/plotter have none; no alternate oracle or direct MD invocation bypass |
| G15 | subprocess spy | exactly the registered serial MD commands can occur inside an authorized image attempt; none occur during consumer stages |

An attempt ID is globally unique within a run.  Atomic attempt records are the
durable source of truth; the JSONL ledger is a unique ordered index of those
records, not an append-anything log.  Each attempt has exactly one `started`
record and at most one terminal record.  A completed attempt has exactly one
terminal record.  Only `failed_before_mean_force` is retryable and costs zero
NoF.  Every status that returned a complete mean-force vector costs one NoF,
even if a separate eligibility gate later fails.

For a fully completed, eligible band:

```text
sum(nof_returned over its distinct attempts) == 20
number of distinct successful logical (band_id, image_index) results == 20
public band cost == 20
```

During an interrupted band, realized attempt NoF may temporarily exceed the
public NoF prefix.  The audit report must expose both quantities.  Resumption
reconciles matching artifacts; it must never erase realized NoF or evaluate an
already returned logical image again.

### H. Resume, replay, and crash consistency

For every crash fixture, compare with an uninterrupted reference run using the
same fake oracle request stream.  After resume, immutable public bytes,
request/path/frame/candidate hashes, controller checkpoint, completion, metric
arrays, movement count, and attempt audit must match.  Time/pid/status metadata
may differ only where explicitly non-scientific.

| ID | injected interruption point | required resume behavior |
|---|---|---|
| H01 | before paid oracle request | rederive identical request; no prior NoF |
| H02 | after an image `started`, before mean force | registered same-seed retry allowed; failed attempt recorded at NoF 0 |
| H03 | after image result artifact, before terminal attempt record | recover artifact once; no rerun |
| H04 | after some or all 20 image artifacts, before complete band return | reuse exact valid artifacts and evaluate only genuinely missing images |
| H05 | after complete band return, before public-state publish | reconcile all 20; publish once without reevaluating |
| H06 | after public validation/HOLD/movement row, before checkpoint replacement | replay row; zero oracle calls for that row; checkpoint catches up |
| H07 | after checkpoint, before status/completion replacement | rebuild non-scientific summary without oracle call |
| H08 | after positive validation publication, before pending checkpoint | replay exact validation and pending candidate; next band is commit, not another validation |
| H09 | after pending commit band artifacts, before target public row | reuse band; publish frozen candidate exactly once |
| H10 | stale checkpoint one row behind public ledger | deterministic replay permitted |
| H11 | checkpoint ahead, more than one row behind, or inconsistent at same index | fail closed |
| H12 | truncated/invalid final public NPZ, atomic record, result, or binding | fail closed; a temporary uncommitted filename may be ignored, never consumed |
| H13 | resume with a stop behind durable NoF or an unregistered larger budget | reject |
| H14 | short registered stop versus longer registered run | common paid prefix and attempt identities are byte-identical |

### I. `jobs=1` and host-global locking

| ID | fixture/action | required result |
|---|---|---|
| I01 | registration has integer `jobs: 1` | accepted |
| I02 | `jobs` is Boolean, zero, two, missing, or CLI-overridden | reject before mutation/oracle |
| I03 | formal oracle provenance/MD command | exactly one serial image loop and GROMACS `ntmpi=1`, `ntomp=1`; no pool/executor |
| I04 | two OS processes use different seeds/parameters/stages but the same canonical Ala lock | first holds lock; second fails immediately with zero artifacts/evaluations |
| I05 | runner attempts a campaign-/seed-/parameter-specific alternate lock | registration/lock-identity test rejects |
| I06 | exception, interrupt, or normal completion | descriptor unlocks in `finally`; existing lock inode is not unlinked as a locking mechanism |
| I07 | lock path is symlink, non-regular, unsafe, or has wrong registered canonical path | fail closed |
| I08 | status metadata says released while another process holds `fcntl` | actual `fcntl` ownership wins; second process still rejects |

The two-process test must use real `fcntl` in subprocesses; mocking `flock`
cannot establish cross-process exclusion.

### J. Acceptance and plotter use exactly the same schema

| ID | fixture/mutation | acceptance | plotter |
|---|---|---|---|
| J01 | valid movement terminal at exact 2000 | same exact row accepted | same exact row plotted |
| J02 | valid HOLD terminal at exact 2000 | same exact row accepted | same exact row plotted |
| J03 | valid pending validation terminal at exact 2000 | source/current row accepted | same source/current row plotted |
| J04 | missing exact checkpoint | reject; no LOCF | reject; no LOCF |
| J05 | legacy state kind/identity | reject | reject |
| J06 | malformed kind/action/pending combination | reject | reject |
| J07 | path or metric mutated after acceptance report | digest mismatch; regenerate only from validated evidence | reject stale report/evidence binding |
| J08 | acceptance report says pass but registration/run/source digest differs | reject | reject |
| J09 | acceptance fails any registered gate | report failure | formal replacement forbidden |
| J10 | preview requested from valid but nonpassing sealed evidence | optional preview may be allowed and visibly marked; formal target remains untouched |
| J11 | formal replacement requested under `DRAFT` | reject | reject before temp/final output |
| J12 | exact contour checkpoints 0, 200, 400, 2000 include HOLD/validation | select exact public rows by NoF | plot those same exact paths |

The shared validator should be tested once with a table of malformed
trajectories, then parametrically assert that both acceptance and plot-data
construction reject every case.  This prevents a permissive plotter from
displaying evidence that the acceptance code would reject, or vice versa.

The acceptance report schema should have one exact V17 formal schema ID.  It
must bind the registration digest, implementation/source digest, each consumed
run manifest and immutable public trajectory digest, the exact checkpoint
grid, the terminal-row indices/kinds, pending-at-horizon disposition, and the
final decision.  The plotter revalidates those bindings before importing
matplotlib.  It must not accept an old V1/V13 analysis ID merely because a
`pass` key is true.

## 4. Property and mutation tests

In addition to the named examples, use generated short ledgers of 1--8 paid
states over the three paid kinds.  Preserve the exact 20-NoF grid and derive
movement count from kind.  For each valid ledger, check:

- replay is deterministic;
- truncating at any public row and resuming produces the same prefix;
- exact checkpoint selection returns that row regardless of kind;
- inserting any zero-NoF refresh changes no public length or NoF;
- changing any one of kind, path, movement flag/count, band ID, NoF, source
  digest, or pending status makes validation fail.

Use deterministic property seeds committed with the test source.  These are
software-test seeds, not formal MD seeds, and must be labelled accordingly.

## 5. Pre-execution acceptance for the implementation itself

No formal MD may start until the following gates have been satisfied in this
order and a new registration has then been sealed prospectively:

1. the r5 NoF-1940 invariant failure is diagnosed in a new controller revision,
   reproduced by a targeted regression, and independently verified as fixed;
2. the registration still says `DRAFT` while code/tests are under review;
3. all unit, mutation, property, two-process lock, and fake-oracle integration
   tests pass under the registered Python runtime;
4. a repository search finds no legacy V13 implementation ID or
   `reparameterized_inner_trial` in the executable R6 acceptance/plot path;
5. the sole formal `evaluate_band` call-site audit passes and the R6 run root
   and attempt ledgers remain absent;
6. final parameter and seed values replace every placeholder, with an explicit
   novelty/prospective attestation made before the first formal attempt;
7. source/input/runtime/Mutagen/host/lock hashes and the full NoF/run order are
   final, independently reviewed, and bound in canonical registration bytes;
8. only then is `status` changed atomically to
   `SEALED_BEFORE_FIRST_FORMAL_MD`, and the runner revalidates the sealed bytes
   immediately before acquiring the host-global Ala lock.

Passing tests does not itself seal the campaign.  A failed test or unresolved
value keeps the draft non-executable; the safe outcome is zero new formal NoF.
