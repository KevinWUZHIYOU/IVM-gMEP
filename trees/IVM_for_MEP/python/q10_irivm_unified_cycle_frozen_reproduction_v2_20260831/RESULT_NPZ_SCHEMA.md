# Frozen-reproduction result NPZ schema

Schema version: `2`.

The writer uses `numpy.savez_compressed` and the reader always uses
`allow_pickle=False`. Unless stated otherwise, integer arrays are `int64`,
floating arrays are `float64`, booleans are `bool`, and text is NumPy Unicode
(`U`), never object dtype. Tangent context codes are `int16` and tangent digest
matrices are `uint8`. Unknown or missing keys fail validation.

Let `N` be the number of internal images, `D=10`, `S` the number of public
states, `T` the number of paid evaluations, `C=5` the number of NoF categories,
`M` the number of main-event JSON records, `R` the number of recovery-event JSON
records, `K` the number of stored recovery states, `P` the number of spring
reparameterization events, `G` the number of tangent events, `J` the number of
tangent contexts, and `H` the number of terminal recovery-memory pairs.

## Identity and immutable scalars

Scalar fields:

```text
schema_version = 2
campaign_id, run_id, config_json
registration_sha256, registration_seal_sha256
algorithm_source_sha256, optimizer_sha256, adapter_sha256
runtime_sha256, backend_sha256, runner_sha256
dependency_manifest_json
exploratory_source_sha256, preformal_equivalence_sha256
instrumentation_equivalence_sha256
runtime_identity_json, optimizer_summary_json
q_sha256
tangent_definition, reparameterization_definition
tangent_transcript_sha256
stop_reason, failure_message
spring_dtau, spring_max_iter, spring_tolerance
budget, terminal_nof
working_equals_public_at_exit, blocked_uncached_misses
tangent_call_count, tangent_mismatch_count, tangent_reparam_call_count
```

`q_matrix` has shape `[10,10]`. JSON text fields must decode to JSON. SHA-256
fields are 64 lowercase hexadecimal characters. `optimizer_sha256` must equal
`algorithm_source_sha256`. The result and its schema-v2 receipt repeat the
sealed algorithm, adapter, runtime, backend, runner, exploratory, preformal,
and instrumentation-equivalence identities.

The frozen execution contract recorded by `runtime_identity_json` is `jobs=1`,
not one BLAS thread: `PYTHONHASHSEED=0`, `OPENBLAS_NUM_THREADS=8`, and
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and
`NUMEXPR_NUM_THREADS` unset. The pre-backend and post-backend
`threadpoolctl` identities are registered separately because loading the QM10
backend adds numerical libraries; each exact identity is checked at its own
boundary, and every loaded numerical pool must report eight threads.

## Public trajectory

```text
public_nf                         [S]
public_outer                      [S]
public_phase                      [S]
public_path_history               [S,N+2,10]
public_energy_history             [S,N+2]
public_gradient_history           [S,N+2,10]
public_state_changed              [S]
```

`public_nf` is strictly increasing, begins at `N`, and ends at
`terminal_nof`. The first and last path images are constant throughout. Every
paid episode ends with a public event at the resulting NoF, including held or
rolled-back episodes.

## Force-evaluation trace and ledger

```text
evaluation_sequence               [T] == 1..T
evaluation_nof                    [T] == 1..T
evaluation_outer                  [T]
evaluation_phase                  [T]
evaluation_image_index            [T], values 1..N
evaluation_coordinate             [T,10]
evaluation_coordinate_sha256      [T]
evaluation_energy                 [T]
evaluation_gradient               [T,10]
optimizer_evaluation_trace_json   [T]
per_image_joint_evaluations       [N+2], endpoints zero
nof_category_names                [C]
nof_category_counts               [C]
```

`T == terminal_nof == sum(per_image_joint_evaluations) ==
sum(nof_category_counts)`. Coordinate hashes are recomputed from canonical
contiguous little-endian float64 coordinate bytes. The category names and order
are frozen as

```text
initial_band
recovery_band_trial
normal_subproblem
direct_spring_check
synchronized_recovery_trial
```

The observer trace, optimizer trace, per-image ledger, and category ledger must
all close to the same `terminal_nof`. One NoF is one uncached joint
energy-gradient evaluation at one internal image. Repeated exact coordinates
for the same internal-image cache are rejected by the independent audit.

## Main and recovery provenance

```text
main_event_json                   [M]
recovery_event_json               [R]
recovery_state_nf                 [K]
recovery_anchor_path              [K,N+2,10]
recovery_candidate_path           [K,N+2,10]
recovery_public_before_path       [K,N+2,10]
```

Event JSON stores scalar online decision quantities and labels; the aligned
numerical recovery paths stay out of JSON. Recovery operations include normal
momentum, normal L-BFGS, secant construction, provisional continuation, and
synchronized normal continuation. They are recovery-subproblem operations,
not reparameterization. Whenever recovery requests reparameterization, it must
enter the sole spring event interface described below.

## Spring reparameterization and tangent provenance

```text
reparam_event_id                  [P] == 1..P
reparam_caller                    [P]
reparam_outer                     [P]
reparam_nof_at_entry              [P]
reparam_input_path                [P,N+2,10]
reparam_fixed_energy              [P,N+2]
reparam_output_path               [P,N+2,10]
reparam_iterations                [P]
reparam_final_residual            [P]
reparam_tangent_call_start        [P]
reparam_tangent_call_count        [P]
tangent_event_id                  [G] == 1..G
tangent_parent_reparam_event_id   [G]
tangent_context_names             [J]
tangent_context_code              [G], int16
tangent_context_counts            [J]
tangent_path_sha256               [G,32], uint8
tangent_energy_sha256             [G,32], uint8
tangent_output_sha256             [G,32], uint8
```

Every operation named reparameterization is exactly the nearest-neighbour
spring flow

```text
L_i = ||x_i - x_{i-1}||
F_i^spring = (L_{i+1} - L_i) tau_i
x_i <- x_i + 0.05 F_i^spring
```

where `tau_i` is the Henkelman--Jónsson improved tangent. The energies in
`reparam_fixed_energy` come from already evaluated source coordinates available
at or before spring entry and remain frozen for all zero-NoF spring iterations.
The raw `reparam_input_path` can be an unevaluated recovery proposal, so its
coordinates are not automatically the energy-source coordinates. The maximum
iteration count is 500, the residual tolerance is `1e-5`, and both endpoints
remain bitwise fixed.
`reparam_tangent_call_count == reparam_iterations` for every event.

`reparam_tangent_call_start` is a zero-based slice offset into the ordered
tangent transcript. Every linked slice must be nonoverlapping and have parent
ID equal to its spring event ID. Tangents outside spring flow use parent ID
`-1`; the coverage vector must show no orphaned or multiply linked spring
tangent.

For each tangent call, path, energy, and output are encoded with the
domain-separated `q10-formal-array-v2` SHA-256 format into three 32-byte rows.
The event ID, signed parent ID, UTF-8 context selected by `tangent_context_code`,
and those three digests are folded in order into
`q10-tangent-transcript-v2`. Context counts must exactly reproduce the code
stream. The write-after adapter independently recomputes the HJ tangent online;
`tangent_mismatch_count` must be zero.

The independent auditor replays every spring event iteration by iteration from
the full input path and frozen energies. Before replay, it must locate every
internal frozen-energy source among paid evaluations at or before
`reparam_nof_at_entry`, independently recompute the source energies, and require
complete prior-paid-source closure. It must not compare the frozen energy vector
to the possibly unevaluated raw input path. It then reproduces every spring
tangent digest, output path, final residual, iteration count, and endpoint
bitwise, without importing the optimizer or runtime. Non-spring tangent calls
retain ordered input/energy/output hashes and are protected by the zero-mismatch
online checker plus the sealed static reachable-call report. Because their full
arrays are intentionally not stored, they are **not** claimed as offline
per-call numerical replays.

## Terminal recovery memory

```text
memory_image_index                [H]
memory_pair_order                 [H]
memory_s                          [H,10]
memory_y                          [H,10]
```

Image indices lie in `1..N`; `(image_index,pair_order)` labels are unique.

## Independent scientific audit

After structural schema validation, the independent auditor recomputes the Q10
potential for every charged coordinate and every public state, verifies the
exact NoF traces and ledgers, and authenticates the registration, source,
dependency, result, and schema-v2 receipt hashes. It independently recomputes
Article-language `I_perp` and the coordinate-wise linear-500 densified energy
barrier.

The shared initial state is checked and excluded. Exact Article public NoF
checkpoints after the initial state give 79 comparisons for N=10 and 98 for
N=20. Candidate values are selected by last observation carried forward
(LOCF). At every checkpoint, both direct float64 inequalities must be strict:

```text
I_perp(IR-IVM) < I_perp(Article IVM)
abs(DeltaE_linear500(IR-IVM) - 67.50188060819488)
  < abs(DeltaE_linear500(Article IVM) - 67.50188060819488)
```

Equality, a missing state, a non-finite value, a checkpoint mismatch, or any
integrity failure is a scientific non-pass.
