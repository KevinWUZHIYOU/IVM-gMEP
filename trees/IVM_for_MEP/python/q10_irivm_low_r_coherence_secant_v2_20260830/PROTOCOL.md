# QM10 low-r coherence-conditioned recovery-memory campaign v2

## Scientific question and status

Can the unified QM10 IR-IVM use only `r < 0.7` and still beat the frozen
Article IVM at every registered latter-half same-Number-of-Forces checkpoint
in both article metrics for both `N_internal=10` and `N_internal=20`?

This campaign is a disclosed post-hoc QM10 method-development confirmation
against a known opponent. It is not an independent transfer, test-set, or
holdout-generalization claim. The optimizer cannot import or read the Article
runs, the reference barrier, either offline metric, the checkpoints, or prior
selection results. Article data are opened only by the separate audit after an
optimization run has finished.

## Why v2 exists and what is (not) changed

The first formal campaign,
`q10-irivm-low-r-coherence-secant-20260830` (v1), executed all four registered
N20 runs. Its registered independent audit rejected two runs with
`first recovery clipped regularized step drifted`. Postmortem reconstruction
identified this as an audit false positive: for a recovery whose raw physical
displacement was at floating-point cancellation scale, the v1 audit recovered
the direction by subtracting two nearly equal coordinates and then used that
ill-conditioned direction to reconstruct the clipped scalar step. This was a
numerical-stability defect in the independent checker, not a change or failure
of the optimizer.

The staged integrity gate behaved conservatively: v1 was reported as an
integrity failure and **did not release any N10 optimization**. The frozen N20
run `685c22444caacc53` (`alpha=1e-3`, `r=0.60`, `m_max=4`) in fact has lower
Article `I_perp` at 51/51 registered checkpoints and lower densified
`Delta E` error at 51/51 registered checkpoints when the raw-step identity is
rechecked without the tiny-displacement cancellation. That scientific result
was not accepted as a v1 formal pass because its registered v1 integrity gate
failed.

V2 is therefore an explicitly post-evidence confirmation, not a holdout. The
optimizer's numerical control flow and arithmetic, recovery, low-r grid, NoF
budget, staged N20-before-N10 rule, metrics, and strict acceptance criterion
are unchanged from v1. The only substantive revision is numerical stabilization
of that independent audit identity, together with a regression test covering
the tiny-displacement case. V2 additionally records the already-computed raw
direction as a logging-only witness before any coordinate addition; this does
not feed back into the algorithm. Campaign identity, the unique host-local
lock, provenance dependencies, and this disclosure are administrative metadata
changes. Narrow AST tests compare the v2 optimizer and runner with v1 and reject
any executable difference beyond those exact metadata/lock changes plus the
one default log field and one direction-copy assignment.

To make the diagnosis and prior observation permanently traceable, v2 freezes
the SHA-256 of these v1 artifacts as preregistration dependencies:

- `registrations/CAMPAIGN_PREREGISTRATION.json`;
- `src/q10_irivm_atomic_tangent_secant_core.py` and `src/audit_results.py`;
- `runs/n20/685c22444caacc53.npz` and its receipt;
- `reports/n20_gate_audit/audit_manifest.json` and `AUDIT_REPORT.md`.

## Fixed non-recovery algorithm

The frozen predecessor is
`q10_irivm_chord_coherence_20260830`. The following components remain
unchanged and are checked by source/AST comparison:

- the QM10 potential, Q transform, endpoints, and common initial paths;
- Henkelman--Jonsson improved tangents for every actual tangent operation;
- the complete-normal proximal Armijo L-BFGS main subproblem;
- the whole-band `I_perp`-monotone main commit or rollback;
- the improved-tangent identical-spring direct reparameterization after every
  outer stage;
- the accepted-history recovery direction, fixed half backtracking, recovery
  projection, whole-band atomic acceptance gate, exact-coordinate cache,
  trust adapter, and global Number-of-Forces ledger.

No central-difference tangent is permitted. Adjacent chords are used only by
the already frozen dimensionless recovery bend diagnostic; they never define
a tangent, normal basis, force projection, or `I_perp`.

## Frozen recovery and the inherited memory rule

Recovery proposal generation, acceptance, and memory publication are exactly
those of v1. Relative to the earlier frozen chord-coherent predecessor, the
already-v1-frozen method behaves as follows. If the evaluated direct
reparameterized band is not acceptable, it:

1. constructs the accepted-history normal L-BFGS raw proposal, with the same
   zero-NoF same-anchor fractions `1`, `1/2`, `1/4`, ... and at most `m_max`
   completed attempts;
2. forms the equal-arclength polyline `P` and improved-spring path `S` without
   force evaluations;
3. at the highest-energy internal image, uses the existing rule

   ```text
   delta_chord = sqrt(1 - dot(u_minus, u_plus))
   strong_horizon = m_max - 1
   strong_horizon * delta_chord <= 1
   ```

   to choose the strong saddle weight `m_max-1`, otherwise the conservative
   weight one;
4. applies the same Henkelman--Jonsson improved-tangent chord-constraint
   projection and evaluates at most one complete projected band; and
5. applies the same whole-band atomic gate. The band must be finite, changed,
   and inside the equidistance ceiling, and it commits only when no worse in
   `I_perp` than the post-main band or when its already evaluated actual
   proximal merit is positive.

The accepted-recovery memory transport inherited from v1 is the only optimizer
revision relative to that earlier chord-coherent predecessor. For every
internal image of a complete accepted recovery, define the physical pair

```text
s_i = x_i_projected - x_i_direct
y_i = g_i_projected - g_i_direct
```

and certify it exactly when

```text
s_i^T y_i > 1e-6 ||s_i|| ||y_i||,
```

with nonzero norms. For every certified pair define the dimensionless
normalized curvature

```text
c_i = (s_i^T y_i) / (||s_i|| ||y_i||).
```

The publication rule is:

- after a strong chord-coherent recovery, publish only the certified pair
  with largest `c_i`; an exact tie is resolved by the smaller image index;
- after a conservative weight-one recovery, publish all certified pairs;
- after the ordinary normal recovery used to exhaust stationarity, publish
  all certified pairs;
- a rejected, incomplete, or interrupted recovery publishes no pair.

The strong recovery is a nonlocal saddle extrapolation, so only its most
directionally coherent measured curvature pair is retained. The conservative
and ordinary recoveries are local, so their complete certified physical
curvature information is retained. This rule uses only the recovery branch
already selected by the existing `m_max` geometry. It adds no tunable control
and reads neither resolution, `r`, alpha, Number of Forces, outer index,
training phase, opponent, nor an offline metric. Ranking and publication are
zero-NoF algebra on energies and gradients already charged for the accepted
projected band.

Each transaction records, separately, the number of curvature-certified
pairs, pairs published, certified pairs omitted by the strong-branch rank,
pairs rejected by curvature, and the published image indices and normalized
curvatures. These counts must close independently in the audit; rank omissions
may not be mislabeled as curvature failures.

## Registered low-r confirmation grid

The same Cartesian grid is registered at both resolutions:

- alpha labels: `1e-3`, `2e-3`;
- `r = [0.60, 0.65]`;
- `m_max = [4]`.

All four settings satisfy `r < 0.7`; all alpha labels have integer mantissas;
and `m_max >= 4`. There are four settings per resolution and eight registered
runs. Parameters never change within a run. The selected controls may differ
between resolutions, but both resolutions execute the same source and the
same registered grid.

Each run has a 3000-NoF budget and a 10000-outer safeguard. Formal execution
is serial (`jobs=1`) under a host-local `fcntl` lock, with `PYTHONHASHSEED=0`
and all registered numerical thread variables equal to one. `N_internal=20`
is run and independently audited first. `N_internal=10` is allowed to start
only after an integrity-valid strict N20 pass exists. Final selection requires
all eight registered results.

The staged commands, executed from the campaign directory with the registered
single-thread environment, are:

```text
python src/run_grid.py --jobs 1 --n-internal 20
python src/audit_results.py --n-internal 20 \
  --output-dir reports/n20_gate_audit
python src/run_grid.py --jobs 1 --n-internal 10
python src/audit_results.py --output-dir reports/result_audit
```

The N10 runner does not merely trust a pre-existing gate manifest. Immediately
before N10 optimization, it reruns the registered audit source with the current
Python interpreter into `reports/n20_gate_audit`, then authenticates the audit
artifacts and all four registered N20 NPZ/receipt identities and NoF ledgers.
It refuses N10 unless that fresh audit is integrity-complete and contains an
N20 strict winner. The SHA-256 of that exact gate manifest is then fixed as the
N10 execution token and stored in every N10 NPZ and receipt; every N20 NPZ and
receipt instead stores the exact token `not_applicable`. The independent final
audit authenticates the gate manifest and its artifacts, then requires each
result and receipt to carry the resolution-appropriate token. The final command
audits all eight members jointly and is the only source for final selection and
the comparison figure.

## Number of Forces

One NoF is one uncached joint energy-gradient evaluation at one internal
image; endpoints are uncharged. Initial, main Armijo, direct reparameterized,
projected recovery, rejected, rolled-back, and partial-budget evaluations are
charged. Exact cache hits, raw recovery construction, polyline and spring
seeding, chord-coherence selection, improved-tangent chord projection,
secant certification, normalized-curvature ranking, and offline metrics cost
zero NoF.

Every transaction stores its per-image direct, raw, projected, recovery, and
total vectors. Raw recovery must be identically zero; recovery must equal the
projected vector; total must equal direct plus recovery; and the complete
per-image ledger must close exactly to terminal NoF.

For raw-step authentication, each transaction initializes
`raw_pre_addition_direction` to `None`. When a positive raw proposal is
constructed, the optimizer copies the complete float64 `directions` array to
that field before computing or adding the candidate displacement. The field is
write-only provenance: it is never read by proposal selection, projection,
acceptance, memory transport, stopping, or NoF accounting. The independent
audit uses this pre-addition witness to verify the regularized step without
recovering a direction from cancellation-prone `raw_path - direct_path`.

The saved audit trail also closes NoF in chronological order, not only in the
aggregate. Every successful charged cache miss appends an ordered record of
its global sequence, outer iteration, phase, image index, and SHA-256 of the
exact contiguous float64 coordinate bytes. Starting from the charged initial
band, the independent audit reconstructs the per-image exact-coordinate cache
from this trace; identifies main, direct-reparameterization, complete projected,
and partial-budget calls; and cross-checks every phase vector. Those independently
reconstructed increments must equal the saved main-stage and direct anchors,
and they alone reconstruct the public-state NoF axis used for LOCF. Thus a
stored stage timestamp or redistributed high-level call count cannot move a
later path to an earlier same-NoF checkpoint while retaining the same final
total.

## Frozen opponent, metrics, and strict acceptance

Frozen Article IVM opponents:

- N10: `paper/ActiveExperiments/muller_q10_10d/image12/data/frozen_runs/tuned_ivm_alpha2e3_selected.npz`;
- N20: `paper/ActiveExperiments/muller_q10_10d/data/frozen_runs/tuned_ivm_alpha1e3_selected.npz`.

The separate audit recomputes:

- article `I_perp` with Henkelman--Jonsson improved tangents;
- the coordinate-wise 500-point linearly densified energy barrier;
- `abs(Delta E - 67.50188060819488)`.

LOCF uses the last complete public state at or below each checkpoint. N10 uses
51 checkpoints from NoF 481 through 951; N20 uses 51 checkpoints from NoF
1199 through 2377. One run passes a resolution only if, at all 51 checkpoints
and by exact floating-point comparison,

```text
I_perp_candidate < I_perp_Article
DeltaE_error_candidate < DeltaE_error_Article.
```

Equality fails. Numerical failures and non-finite trajectories are ineligible.
The campaign passes only if both resolutions have a strict passing run.

## Disclosed pre-registration method-development evidence

Temporary `/private/tmp` prototypes were used to derive this rule. They
bypassed the formal parameter/registration gates, wrote no artifact inside
this campaign, and read Article metrics only after optimization. The unified
prototype produced the following exploratory results:

- N20, `alpha=1e-3`, `r=0.60`, `m_max=4`: both metrics lower at 51/51
  checkpoints, terminal NoF 2276; all accepted normal recoveries selected the
  strong branch and published one ranked pair;
- N10, `alpha=2e-3`, `r=0.65`, `m_max=4`: both metrics lower at 51/51
  checkpoints, terminal NoF 847; all accepted normal recoveries selected the
  conservative branch and published all certified pairs.

Each exploratory case was repeated independently and reproduced its complete
trajectory hash. This evidence motivated the frozen confirmation grid; it is
not a formal result and makes the forthcoming campaign a reproducibility
confirmation rather than a holdout test.

## Registration and audit gate

`registrations/CAMPAIGN_PREREGISTRATION.json` must exist and freeze the core,
runner, independent audit, tests, protocol, backend, metric, frozen opponents,
the disclosed v1 registration/core/audit/N20 evidence and failed gate report,
and all other dependencies by SHA-256 before the first formal result, receipt,
or execution report. The runner must refuse unregistered controls, non-serial
execution, missing single-thread settings, a busy v2-specific host-local lock,
drifted hashes, or overwrite.

The independent audit must reopen every frozen dependency; recompute stored
energies and gradients; reconstruct every recovery seed, atomic gate, NoF
vector, curvature-certified set, normalized-curvature ranking, tie break, and
publication count/index/score; authenticate the outer-start, post-main, and
committed-public-path state chain; reconstruct the public NoF trajectory from
the ordered exact-coordinate miss trace; independently replay the registered
improved-tangent chord-Newton projection; verify raw-step, regularized-prediction,
trust-radius, and same-anchor half-backtracking algebra; authenticate the
termination condition and staged N20-to-N10 manifest token; recompute both
article metrics and all 51 LOCF comparisons; and report integrity failure
separately from a valid scientific non-pass.

The registered core, runner, and their AST/source hashes are the execution
trust boundary. The offline audit verifies the saved raw-step algebra and all
downstream geometry, force evaluations, acceptance, state, NoF, and metric
claims, but it is not a cryptographic execution attestation and does not
independently regenerate every accepted main-stage L-BFGS memory pair from
scratch. This limitation is explicit and does not weaken reproducibility: the
same registered optimizer is rerunnable, and its complete formal artifacts are
source- and dependency-pinned before execution.
