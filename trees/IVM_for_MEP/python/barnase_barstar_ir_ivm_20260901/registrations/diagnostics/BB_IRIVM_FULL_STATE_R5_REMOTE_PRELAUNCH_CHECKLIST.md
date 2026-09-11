# BB IR-IVM full-state R5 remote prelaunch checklist

Status: `PREPARED_ONLY_DO_NOT_LAUNCH_WITH_PLACEHOLDERS`

This checklist is zero-OpenMM preparation. It does not authorize a physical
run. Replace every angle-bracket token only after the R5 recorder, runner,
verifier, registration, and one-use authorization have been frozen under new
paths and hashes. Do not reuse any R2--R4 label, authorization, output, WAL, or
partial shard.

## Required frozen facts

| Gate | Required observation |
|---|---|
| Mutagen | Dedicated session `bb-irivm-20260901` is Connected/Watching and a fresh flush succeeds |
| source identity | Remote SHA-256 values exactly equal the R5 registration bindings, including all selected algorithm sources |
| authorization | New one-use R5 authorization directly binds the final R5 registration and runtime argv hash |
| runtime delta | Relative to the selected NF9990 run, only the frozen output-only differences are present; all numerical controls and both NF budgets are unchanged |
| preflight | R5 zero-OpenMM preflight returns PASS and explicitly reports no OpenMM import/context/call |
| locks | Both canonical host-local BB locks can be acquired together non-blockingly before launch |
| threads | `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `OPENMM_CPU_THREADS` all equal `1`; jobs equals `1` |
| disk | Free bytes satisfy the minimum frozen in R5 registration/authorization |
| clean output | New R5 result directory, observer directory, result, audit, manifest, WAL, and launch receipt do not exist |
| supersession | R2 user-termination receipt and R2--R4 failure/supersession receipts are present and hash-bound; none of their outputs is imported |

## Exact sequence before any launch

Run the first two commands on the local Mac:

```bash
/Users/wuzhiyou/.local/bin/mutagen sync list
/Users/wuzhiyou/.local/bin/mutagen sync flush bb-irivm-20260901
```

The listing must show the dedicated session with both Alpha and Beta connected
and status Watching. A paused generic Mutagen project is not a substitute.

After the flush, compare the complete frozen inventory on the remote Mac. The
expected SHA list must come from the immutable R5 registration, not from a
manually maintained second list:

```bash
ssh -i /Users/wuzhiyou/.ssh/ivm_mac_mini_ed25519 \
  -o BatchMode=yes -o ConnectTimeout=8 \
  -o UserKnownHostsFile=/Users/wuzhiyou/.ssh/known_hosts \
  -o StrictHostKeyChecking=yes wuzhiyou@100.74.85.3 \
  'shasum -a 256 <ABS_R5_RECORDER> <ABS_R5_RUNNER> <ABS_R5_VERIFIER> \
    <ABS_R5_TEST> <ABS_R5_REGISTRATION> <ABS_R5_AUTHORIZATION> \
    /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/src/bb_core.py \
    /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/src/bb_ffprecon.py \
    /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/src/storage_policy.py \
    /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/src/run_ivm_v9_spring_projected_production_20260730.py \
    /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/src/frozen_discrete_b040.py \
    /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/src/run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py \
    /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/src/run_ivm_v9_discrete_selected_eta_shim.py \
    /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/src/run_ivm_v9_discrete_ffrbk_takeover_grid_shim.py \
    /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/src/run_ivm_v9_total_internal_nf_budget.py \
    /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/src/run_barnase_barstar_ir_ivm_member.py'
```

Every emitted digest must equal the immutable R5 registration/authorization
inventory. The later R5 preflight must independently fail closed on any source,
argv, authorization, or supersession mismatch and must not import OpenMM.

Probe both canonical locks together and the frozen disk reserve using the
already audited host-local lock wrapper. The receipt path must be new:

```bash
ssh -i /Users/wuzhiyou/.ssh/ivm_mac_mini_ed25519 \
  -o BatchMode=yes -o ConnectTimeout=8 \
  -o UserKnownHostsFile=/Users/wuzhiyou/.ssh/known_hosts \
  -o StrictHostKeyChecking=yes wuzhiyou@100.74.85.3 \
  'cd /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901 && \
   /Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
     scripts/with_host_local_bb_locks.py \
     --receipt <ABS_NEW_R5_LOCK_PROBE_RECEIPT> \
     --min-free-gib <R5_MINIMUM_FREE_GIB> -- /usr/bin/true'
```

Run the R5 preflight with all six caps explicitly set. It must probe/validate
the same two canonical locks, confirm `jobs=1`, check the actual byte reserve,
and reject every pre-existing R5 output:

```bash
ssh -i /Users/wuzhiyou/.ssh/ivm_mac_mini_ed25519 \
  -o BatchMode=yes -o ConnectTimeout=8 \
  -o UserKnownHostsFile=/Users/wuzhiyou/.ssh/known_hosts \
  -o StrictHostKeyChecking=yes wuzhiyou@100.74.85.3 \
  'cd /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901 && \
   env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENMM_CPU_THREADS=1 \
   /Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python <ABS_R5_RUNNER> \
     --root /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901 \
     preflight --registration <ABS_R5_REGISTRATION> \
     --authorization <ABS_R5_AUTHORIZATION>'
```

Before approving a launch, archive a local prelaunch receipt that binds:

- Mutagen session name and flush time;
- remote source, registration, authorization, runner, recorder, verifier, and
  test hashes;
- exact runtime argv and its SHA-256;
- the zero-OpenMM preflight output;
- the two-lock probe receipt and both canonical lock paths;
- six thread caps, jobs, free bytes, and required minimum;
- the negative inventory proving no prior R5 output exists;
- the R2 user-termination receipt plus R2--R4 failure receipts.

Only after this checklist has no placeholders and the resulting prelaunch
receipt is independently verified may a separate launch command be prepared.
This document intentionally contains no physical launch command.
