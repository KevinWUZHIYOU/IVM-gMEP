# Registration boundary

This directory is reserved for the canonical
`CAMPAIGN_PREREGISTRATION.json` and its exact SHA-256 seal. Before those two
files exist and agree byte-for-byte, no optimizer output belongs to the formal
campaign. Once installed, neither file may be replaced or repaired in place.

The registration's campaign identity is
`q10-irivm-unified-cycle-frozen-reproduction-v2-20260831`. It supersedes v1
only because the v1 independent auditor associated each spring's frozen energy
vector with the unevaluated raw proposal path instead of with the actual
previously paid anchor/direct/public/recovery source coordinates. The v1
registration, results, and failure remain immutable in their original campaign
directory. V2 does not change the optimizer, adapter, controls, runtime, or
expected trajectories. Its unchanged run IDs and relative output names are
isolated under the new v2 campaign root.

The campaign is a post-selection frozen reproduction on the same QM10 data,
metrics, and Article-IVM opponent already inspected during tuning. The
registration must say this explicitly and must not label the experiment as a
holdout, transfer, prospective selection, or generalization test.

## Required registered members and acceptance rule

Exactly two members are frozen, with one shared source and no post-run winner
selection:

| N_int | alpha | r | m_max | Article checkpoints after initial |
|---:|---:|---:|---:|---:|
| 10 | `3e-3` | 0.1 | 7 | 79 |
| 20 | `1e-3` | 0.3 | 8 | 98 |

Each member has a 3000-NoF budget. The common initial state is authenticated
and excluded. At every exact Article checkpoint after the initial state, LOCF
is applied and both strict float64 comparisons must hold:

```text
I_perp(IR-IVM) < I_perp(Article IVM)
abs(DeltaE_linear500(IR-IVM) - 67.50188060819488)
  < abs(DeltaE_linear500(Article IVM) - 67.50188060819488)
```

A registered failure is reported; it is not replaced or retuned.

## Algorithm identity that must be frozen

The N=10 and N=20 members execute the same source and branch formulas. Only
their registered `alpha`, `r`, and `m_max` differ. The registration must freeze
the sole reparameterization as

```text
F_i^spring = (L_{i+1} - L_i) tau_i,
```

where `tau_i` is the Henkelman--Jónsson improved tangent. It must also freeze
`dtau=0.05`, `spring_max_iter=500`, `spring_tolerance=1e-5`, evaluated energies
held fixed throughout each spring event, and fixed endpoints. Normal momentum,
normal L-BFGS, secants, provisional continuation, and synchronized normal
continuation are recovery operations, not reparameterization. Central,
averaged-chord, polyline, cubic, saddle, and equal-arclength
reparameterizations are forbidden.

## Files, inputs, identities, and environment

Before sealing, the registration must:

1. Hash every campaign source in `source_sha256`, including the optimizer,
   adapter, schema, runner, static checks, independent auditor, plotter, and
   all evidence reports.
2. Hash every repository input/dependency in `dependency_sha256`, using
   `repo:` paths. `registered_inputs` must explicitly identify the metric
   source, Article N=10 and N=20 archives, and both frozen Q archives.
3. Provide `result_identity` and `result_identity_paths` for
   `algorithm_source_sha256`, `optimizer_sha256`, `adapter_sha256`,
   `runtime_sha256`, `backend_sha256`, `runner_sha256`,
   `exploratory_source_sha256`, `preformal_equivalence_sha256`, and
   `instrumentation_equivalence_sha256`, plus the frozen Q array digest.
4. Record `jobs=1` and the actual frozen numerical environment:
   `PYTHONHASHSEED=0`, `OPENBLAS_NUM_THREADS=8`, with `OMP_NUM_THREADS`,
   `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `NUMEXPR_NUM_THREADS`
   unset. It must register the exact pre-backend and post-backend
   `threadpoolctl` identities separately and require every loaded numerical
   pool to report eight threads.
5. Freeze schema version 2, the five closed NoF categories, the one-NoF unit,
   and the full independent-audit and figure gates.

## Evidence and empty-output gate

The registration may be written and sealed only after all of the following
pass:

- exploratory-to-materialized equivalence;
- fresh N=10/N=20 write-after instrumentation equivalence;
- the static reachable-call contract, including sole HJ improved tangent and
  sole nearest-neighbour spring reparameterization;
- the canonical v1-auditor-failure diagnosis, including exact old
  registration/result hashes, `algorithm_changed_in_v2=false`, and complete
  prior-paid-source closure for all 134 N=10 and 188 N=20 spring events;
- schema/adversarial/audit/plot tests under Python 3.8;
- complete source, dependency, input, and runtime hashes; and
- confirmation that `runs/n10/`, `runs/n20/`, and both receipt directories are
  empty.

The SHA-256 of `evidence/v1_auditor_failure_diagnosis.json` must be included in
`preformal_evidence`. The diagnosis is a gate, not a replacement for the v2
auditor: the auditor must independently reconstruct each frozen-energy source
at or before the spring-entry NoF, require every internal source coordinate in
the prior paid trace, recompute those energies, and then replay the spring from
its raw input path using the authenticated frozen values.

The seal is the exact SHA-256 of canonical JSON bytes. After sealing, N=20 and
N=10 run serially under one host-local `fcntl` lock. Each installed NPZ is
reopened and schema-validated before its schema-v2 receipt is written. Any
post-seal drift in a source, evidence file, dependency, input, control,
environment, result, or receipt requires a new campaign ID; it is never patched
in place.

## Independent-audit proof boundary

The auditor independently recomputes all charged and public-state Q10
energies/gradients, NoF closure, all spring iterations and their HJ tangents,
Article metrics, exact checkpoints, LOCF, and both strict inequalities. For
non-spring tangents it verifies the ordered v2 digest transcript, zero online
HJ mismatches, and the sealed static reachable-call proof. Because their full
arrays are not stored, the registration and report must not claim that every
non-spring tangent was independently replayed offline.

For spring energies, "independently recomputes" means recomputing the actual
previously paid source coordinates, not assuming that the frozen energy vector
was evaluated at the possibly unevaluated raw recovery proposal passed into
the zero-NoF spring.
