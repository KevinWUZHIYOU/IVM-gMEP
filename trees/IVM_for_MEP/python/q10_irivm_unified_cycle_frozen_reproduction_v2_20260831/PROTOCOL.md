# QM10 unified spring-cycle IR-IVM frozen-reproduction protocol

This is the protocol for the designated formal campaign directory. Before a
canonical preregistration and matching SHA-256 seal exist, the directory is
only pre-registration material and must not be used to claim a formal result.
After sealing, the immutable registration—not this status paragraph—defines
the formal boundary.

Campaign ID:
`q10-irivm-unified-cycle-frozen-reproduction-v2-20260831`.

This campaign supersedes
`q10-irivm-unified-cycle-frozen-reproduction-20260831` only at the independent
audit layer. The v1 registration, results, and failed audit are preserved in
their original campaign directory. The optimizer, write-after adapter,
controls, numerical runtime, and expected trajectories are unchanged in v2;
in particular, the optimizer and adapter bytes are identical to v1. The same
control-derived run IDs therefore remain valid, but all v2 results and receipts
resolve beneath this new v2 campaign root:

```text
python/q10_irivm_unified_cycle_frozen_reproduction_v2_20260831/
  runs/n10/0fa57295d005c597.npz
  runs/n20/8cb8b3a582ac6f8e.npz
  reports/receipts/n10/0fa57295d005c597.receipt.json
  reports/receipts/n20/8cb8b3a582ac6f8e.receipt.json
```

## Claim boundary

This is a **post-selection frozen reproduction** after exploratory tuning on
QM10 and inspection of the same QM10 Article-IVM opponent curves and metrics.
The parameters, acceptance rule, and comparison opponent were selected using
this same example. It is not a holdout, transfer, prospective model-selection,
or generalization test. The only permitted formal claim is that two source- and
parameter-frozen runs reproduce the already observed full-trajectory,
same-Number-of-Forces comparison under a preregistered execution, accounting,
metric, and audit protocol.

Exactly two members will be registered. There is no formal grid and no
post-run winner selection:

| Internal images | alpha | r | m_max | Article checkpoints |
|---:|---:|---:|---:|---:|
| 10 | `3e-3` | 0.1 | 7 | 79 |
| 20 | `1e-3` | 0.3 | 8 | 98 |

The budget is 3000 Number of Forces (NoF) for each member. A failed member is
reported as a valid scientific non-pass if its integrity audit succeeds; it is
never replaced within the sealed campaign.

## Frozen algorithm contract

The N=10 and N=20 members use the same source and the same branch formulas.
Resolution may determine array sizes and complete-band budget feasibility, but
must not select a different method. The optimizer cannot read Article-IVM data,
offline metrics, the reference barrier, registered checkpoint arrays, or any
post-hoc score.

Only `alpha`, `r`, and `m_max` are tuned controls. The final registration must
enumerate every other fixed numerical constant and hash the exact source that
uses it. No control may depend on NoF, outer iteration, a named training phase,
or the Article trajectory.

### Tangent and reparameterization

Every reachable tangent operation uses the Henkelman--Jónsson improved
tangent. Every operation called reparameterization uses the sole
nearest-neighbour spring-flow entry point and exactly

```text
L_i = ||x_i - x_{i-1}||
F_i^spring = (L_{i+1} - L_i) tau_i
x_i <- x_i + dtau F_i^spring
```

Here `tau_i` is the Henkelman--Jónsson improved tangent, `dtau=0.05`, the
maximum number of spring iterations is 500, and the stopping tolerance on the
maximum spring-force norm is `1e-5`. The evaluated image energies supplied at
spring entry are frozen throughout that spring event: moving the images inside
the zero-NoF spring loop does not trigger new potential evaluations. Both
endpoints are restored bitwise and remain fixed. Spring algebra and tangent
construction are zero-NoF operations.

Central-difference, averaged-chord, polyline, cubic, saddle, and equal-arclength
reparameterizations are forbidden. Normal momentum, normal L-BFGS, secant
construction, provisional continuation, and synchronized normal continuation
are recovery-subproblem operations, not reparameterization. If a recovery path
is reparameterized, it must call the same sole HJ spring-flow entry point and
log a distinct spring event. No recovery construction may be relabelled as
reparameterization.

The N=10 and N=20 runs execute this same source and these same branch formulas.
They differ only in path size and their registered `alpha`, `r`, and `m_max`.

## Number of Forces

One NoF is one uncached joint energy-gradient evaluation at one internal image.
Endpoint evaluations are not charged. The initial path costs `N_internal`.
Every cache miss is charged, including main trials, recovery trials, rejected
private/provisional trials, and rolled-back trials. Exact-coordinate cache hits
are free. Directions, predictions, tangents, spring flow, and offline metrics
are free. The budget is a ceiling; an action requiring a complete internal
image band may be refused when the remaining budget is smaller than that band.

The result stores the full ordered miss trace, per-image counts, and category
counts. Independently audited closure requires

```text
trace length = terminal NoF
             = sum(per-image joint evaluations)
             = sum(NoF category counts).
```

## Registered acceptance rule

The shared initial state is verified but excluded. Checkpoints are exactly the
frozen Article public NoF values after the initial entry: 79 for N=10 and 98
for N=20. At each checkpoint, last observation carried forward (LOCF) is used
for both methods. Direct float64 comparisons must satisfy both inequalities:

```text
I_perp(IR-IVM) < I_perp(Article IVM)
abs(DeltaE_linear500(IR-IVM) - 67.50188060819488)
  < abs(DeltaE_linear500(Article IVM) - 67.50188060819488)
```

Equality, missing state, non-finite data, changed checkpoints, or any integrity
failure fails. Public `I_perp` rise count and maximum rise are reported as
secondary diagnostics; they are not silently inserted into or removed from the
registered primary rule.

## Result, audit, and figure boundary

The exact NPZ contract is specified in `RESULT_NPZ_SCHEMA.md` and implemented
by `src/result_schema.py`. It uses no pickle/object arrays. The independent
auditor must not import the optimizer or its numerical runtime. It separately
checks hashes, NoF accounting, every charged potential evaluation, every
public state, every spring-flow event and every HJ tangent used inside each
spring iteration, the two metrics, the exact Article checkpoints, LOCF, and
both strict inequalities.

The proof boundary for tangents outside spring flow is narrower and must remain
explicit. Their full numerical path/energy/output arrays are not stored. Each
call instead retains three domain-separated SHA-256 digests, its ordered event
ID, context code, and parent code; a transcript hash closes the sequence. A
write-after observer also recomputes the HJ tangent online and requires zero
mismatches, while a sealed static reachable-call report proves that the only
reachable tangent target is `improved_tangents`. These facts authenticate and
guard every non-spring tangent call, but they are not described as an offline
per-call numerical replay. In contrast, all spring tangents are independently
reconstructed iteration by iteration from full stored spring inputs.

The comparison figure may be generated only from hash-authenticated audit
tables after a complete scientific pass. It is a 2x2 full-trajectory figure:
columns N=10/N=20 and rows `I_perp`/densified barrier absolute error, with
`Article IVM` and `IR-IVM` as the method names. The x axis is `Number of
Forces`; no training-stage shading is allowed.

### Why v2 supersedes the v1 audit

The v1 auditor rejected the first spring event because it compared the stored
frozen energy vector with energies recomputed at the unevaluated raw
reparameterization input path. That comparison was invalid. A zero-NoF spring
may start from a recovery proposal while using energies that were already paid
for at its anchor, direct state, public state, or evaluated recovery candidate;
the frozen energies are not claimed to be evaluations of every coordinate in
the raw proposal path.

The v2 auditor must locate, at or before each spring event's entry NoF, the
actual previously paid source coordinate for every internal frozen energy. It
then independently recomputes every source energy, requires every source
coordinate to occur in the prior paid trace, and only then replays the spring
from `reparam_input_path` plus that authenticated frozen energy vector. The
canonical preformal diagnosis closes all 134 N=10 and all 188 N=20 spring
events with zero fixed-energy error. Its old registration/result identities,
the two-resolution prior-paid-source closure, and the assertion
`algorithm_changed_in_v2=false` are mandatory preregistration gates, and the
diagnosis file's SHA-256 is recorded in `preformal_evidence`.

This is an auditor provenance correction, not an algorithm change and not a
new tuning opportunity. Any v2 claim must state that distinction explicitly.

## Formal sequence and fail-closed gates

1. Preserve and hash the exact exploratory candidate that produced the claimed
   trajectories.
2. Materialize the portable optimizer without changing numerical order. Keep
   the historical broad numerical core hash-pinned, and prove by static
   reachability that its dormant legacy geometry is not reachable from
   `run_clean`; produce a two-resolution bitwise preformal equivalence record.
3. Add write-only audit logging and prove it trajectory-equivalent.
4. Run static contract and adversarial independent-audit tests.
5. Authenticate the preserved v1 audit failure and independently reconstructed
   prior-paid energy sources for all N=10 and N=20 spring events.
6. Fill every preregistration field and hash. Confirm formal run/receipt
   directories contain no member artifact. Atomically seal the canonical JSON
   before the first formal optimizer invocation.
7. Under one nonblocking host-local `fcntl` lock and `jobs=1`, run the fixed
   N=20 then N=10 members serially. The frozen numerical environment is
   `PYTHONHASHSEED=0`, `OPENBLAS_NUM_THREADS=8`, with `OMP_NUM_THREADS`,
   `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `NUMEXPR_NUM_THREADS`
   unset. The distinct pre-backend and post-backend `threadpoolctl` identities
   are each frozen exactly; every loaded numerical pool must report eight
   threads. Atomically save, reopen, and validate each NPZ before issuing its
   schema-v2 receipt.
8. Run the independent audit. Do not tune or patch a failed member.
9. Only after integrity completion and scientific pass, draw and visually
   inspect the registered figure and write provenance.

Any post-seal drift in source, dependency, controls, constants, environment,
inputs, metrics, result, or receipt requires a new campaign ID. It is never
repaired in place.

## Pre-registration boundary

- The materialized optimizer, write-after adapter, schema-v2 runner, independent
  auditor, and plotting code are present, but their final bytes and all
  repository dependencies still have to be hash-frozen together.
- The sealed static reachable-call report and the fresh two-resolution
  instrumentation-equivalence report must be generated and placed under
  `evidence/` before registration.
- Runtime identity, OpenBLAS/threadpool identity, registered inputs, source
  hashes, dependency hashes, and result-identity paths must be filled into one
  canonical registration and sealed while formal run/receipt directories are
  empty.
- Before sealing, no result, receipt, audit report, or figure is a formal
  artifact. The one-shot registration command must refuse to overwrite an
  existing seal or to proceed when any formal result/receipt already exists.
