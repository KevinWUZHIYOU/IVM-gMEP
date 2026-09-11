# Barnase--Barstar IR-IVM transfer

This directory contains the registered Barnase--Barstar transfer of the
successful Chignolin IR-IVM and its paid optimization results.  It is
deliberately the **central-tangent Chignolin branch**, not the later
Henkelman--Jónsson-tangent unified branch.  The initial zero-Force construction
gates have passed; subsequent physical runs and diagnostics are recorded by
immutable member registrations, execution authorizations, result audits, and
metric contracts rather than inferred from filenames.

## What is frozen

The numerical core `src/frozen_discrete_b040.py`, selector shim, and selected
eta shim are byte-for-byte copies of the Chignolin sources.  Each frozen outer
subproblem uses normalized central differences, the same FFrbk-shifted H0 and
H0-initialized L-BFGS candidates, capped predicted-decrease selection, the
one-way L-BFGS takeover latch, eta=1.1, history 7, the 1 Angstrom whole-band
atom-displacement cap, residual ratio `r`, and `m_max` fully evaluated inner
steps.  The state-curve reparameterization is mandatory every outer iteration,
including its direct proposal, per-image objective acceptance, and recovery
transaction.  Rejected direct and recovery evaluations remain charged.

Only system necessities changed: 32 images (30 internal), 3132 atoms/9396
Cartesian coordinates, the canonical BB topology/path, and the exact formal
BB CPU/double Amber14/GBn2 oracle.  The sparse FFrbk matrices still rebuild at
the same places as Chignolin; `src/bb_ffprecon.py` only removes algebraically
zero nonperiodic wrapping work.  It does not call OpenMM or alter a matrix,
solve, direction, or acceptance rule.

The precise allowlist and forbidden algorithm drift are in
`contracts/MIGRATION_CONTRACT.json`.

## Inputs and execution gates

`inputs/INPUT_POINTERS.json` records the formal 32 x 9396 initial path, PDB,
and inherited endpoint joint E/F provenance.  The endpoint values are extracted
without OpenMM from an already audited canonical dense200 result; its data file,
diagnostic audit, coordinates, energies, and forces are all independently
hash-bound.  `scripts/stage_canonical_inputs.py` stages these inputs without an
OpenMM call; existing files are never overwritten and all file/array hashes are
checked.

`registrations/CAMPAIGN_PREREGISTRATION.json` is construction-only.  The member
adapter supports `--register-only`, but refuses to enter the runtime unless an
immutable `registrations/execution_authorizations/{label}.json` binds that
exact label plus the SHA-256 hashes of its member registration, the campaign
preregistration, and the migration contract.  Issued authorizations are
one-member, non-resumed, and non-overwriting.  The smoke member and the five
retained 480-`N_F` alpha-scan members have such authorizations.  Values above
`alpha=1e3` were cancelled with zero optimization `N_F` under
`registrations/USER_ALPHA_SCOPE_AMENDMENT_20260901_R2.json` and have no active
authorization path.

The batch runner is strictly serial (`jobs=1`).  It acquires both the canonical
and Downloads `.bb_global_heavy_task.lock` files in sorted order with
`fcntl(LOCK_EX|LOCK_NB)`, so the two checkouts cannot each launch a heavy BB
member.  After locking it fails closed unless at least 2 GiB is free for a
smoke/screen member or 8 GiB for production, and atomically records the
observed free bytes and threshold.  On the remote Mac the batch must still be
hosted by the repository's `scripts/run-detached.zsh` protocol after Mutagen
gates pass.

## Number of Forces

The primary `Number of Forces` is every physical joint OpenMM `getState`
attempt made by this optimization member.  The two fixed endpoints are common
already-computed data: their hash-verified E/F values are preloaded before the
first path query, so both endpoint cache misses and endpoint `getState` attempts
are zero.  Thus `Number of Forces` is exactly the internal-image miss count and
a full member has `optimization_internal_nf = Number of Forces <= B`, where
the registered budget `B` is a hard upper bound rather than a target that may
be overshot.  Equality is valid.  The only valid strict inequality is the
registered atomic-remainder case: the next complete 30-internal-image path
evaluation is refused when `0 < B - Number of Forces < 30`, all physical
attempt/evaluation/miss ledgers agree, no endpoint is charged, and the paid
terminal partial transaction is quarantined.  A failed internal `getState`
attempt still counts; any such member is invalid/failed and its logs must be
preserved.  Any other early stop is likewise ineligible.  These rules are
operative in
`registrations/USER_BUDGET_ATOMICITY_AMENDMENT_20260901_R1.json`.

A stored path is labelled by the cumulative physical `Number of Forces` at
which that fully reparameterization-committed path became available.  Paid
trial or recovery evaluations after the last such commit remain charged to the
member total, but they are not assigned to an uncommitted path or used as a
metric point.

## Storage and resource risk

One BB path is about 2.4 MB before container overhead.  The upstream
`--save-all-paths` behavior can therefore create hundreds of MB per member.
The explicit `--path-storage-mode committed` option retains only the initial,
fully reparameterization-committed, and final paths.  The online runtime never
reads this archive; the synthetic prefix test proves full/thin online states
and final state are identical.  `legacy` remains the default so storage is not
silently changed.

The 30 independent 9396-dimensional FFrbk matrices and shifted sparse solves
are the principal memory/wall-time risk.  Run only one BB member per device;
start with one FFrbk build worker.  Full one-image standard/fast matrix parity
can be checked later, without a Force call, using
`scripts/verify_ff_builder_parity_zero_force.py`.  That heavier check is not
part of the lightweight suite; it also acquires both BB locks because System
construction and the large matrices compete for the same host resources.

## Executed search and principal diagnostics

The 240-`N_F` smoke member passed the exact internal-image ledger in
`reports/gates/SMOKE_240_GATE.json`.  At fixed `r=0.5` and `m_max=5`, the
retained controlled scan

```text
alpha in {1e-1, 1e0, 1e1, 1e2, 1e3}
```

has completed at a registered budget of 480 optimization `N_F` per member.
Each successful audit records exactly 480 physical joint energy/force
evaluations on the 30 internal images, 16 per internal image, and zero fixed
endpoint evaluations.  Only fully closed, equal-arclength-reparameterized path
states are eligible; a charged terminal partial transaction is quarantined.

The operative pre-selection metric contract is
`registrations/diagnostics/BB_IRIVM_ALPHA_PRINCIPAL_DIAGNOSTICS_20260901_R2.json`.
It froze every eligible path before selection.  Every path is then sampled at
exactly 500 points, including both endpoints, uniformly in unweighted Cartesian
arclength along its piecewise-linear 32-image chain.  The two principal
quantities are the discrete, unnormalized normal residual `I_perp` and the
densified 0 K forward barrier `Delta E`.  Densification requests energy only:
its force-request count and contribution to optimization `N_F` are both zero.
The online sparse-image barrier is a runtime health observation and is not used
for parameter ranking or figures.

After the densified ledger closes, the pure post-processing selector writes an
immutable receipt containing both raw quantities, their same-state Pareto
relation, and the three alpha values admitted to independent 990-`N_F`
confirmation runs.  `scripts/issue_execution_authorization.py` rejects any
alpha-confirmation member that is not named by that receipt.

## Zero-Force verification

The lightweight suite imports no OpenMM Context and uses only synthetic
quadratics/fake ledgers plus immutable input hashing:

The environment does not require `pytest`.  The directly executable unit-test
modules used by the current gates include:

```bash
PYTHONPATH=src:scripts /Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python tests/test_discrete_iperp_densified_barrier.py
PYTHONPATH=scripts /Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python tests/test_execution_authorization.py
```

Offline comparison metrics, once members exist, must use the same
arc-length-uniform 500-point densification and diagnostic oracle for every
eligible IR-IVM and ODE-NEB path.  Those diagnostic calls are separate from
optimization NoF and may not feed back into tuning.

## Final-figure traceability

`FINAL_FIGURES_TRACEABILITY_TEMPLATE.md` remains the inert, fail-closed
checklist.  The completed, placeholder-free registration is
`FINAL_FIGURES_TRACEABILITY_20260902.md`.  It binds the selected member, the
4056/9985 NoF decompositions, the exact-prefix gate, uniform arc-length-500
diagnostics, both chronological targeted ODE--NEB rounds, plotting programs,
same-stem machine provenance, rendering checks, and reproduction entry points
for the two active-candidate PDFs.  The inert template is preserved unchanged
so it cannot be mistaken for a result.
