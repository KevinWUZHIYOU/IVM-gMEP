# V17 formal campaign activation plan — DRAFT / DO NOT EXECUTE

## Current status

- Core readiness: blocked.  The inherited V17 revision failed in r5 at seed
  1701, trajectory ordinal 104, incremental NoF 1940 with `an outer with
  completed movements must have anchor bounds`; r6 contains no fix or targeted
  real-prefix regression.
- Candidate: pending the complete r5 288-trajectory result and its frozen
  selection receipt.
- Formal optimization seeds: pending prospective selection and novelty
  attestation.
- Terminal-diagnostic, bootstrap, and contour seeds: pending prospective
  selection and novelty attestation.
- MD/reference inputs and runtime hashes: pending final seal.
- Formal MD authorization: false.
- Formal evidence: false.
- Formal run root: intentionally absent.
- Formal PDF modification: forbidden.

## Work packages

1. **Core defect gate** — diagnose the r5 NoF-1940 invariant failure in a new
   controller revision, preserve the failing prefix as a regression fixture or
   content-addressed replay input, prove exact resume/accounting, and obtain an
   independent review.  Do not patch or relabel the frozen r5 source.
2. **Development result gate** — rerun or continue only under a newly sealed
   development revision, verify exactly 288 trajectories, freeze the complete
   analysis, and apply its pre-existing selection rule once.
3. **Candidate receipt** — record one V17 parameter combination and bind the r5
   plan, manifest, ledgers, analyzer, analysis, and selection receipt hashes.
4. **Prospective randomness** — after candidate selection, register three new
   optimization seeds, ten new paired diagnostic seeds, one new bootstrap seed,
   and one contour seed chosen from the optimization seeds.  Attest nonreuse.
5. **Input/runtime seal** — bind the causal shared initial observation, actual
   current-IVM history, periodic FES, FENEB reference, MD inputs, GROMACS,
   PLUMED, Mutagen, source files, tests, and unchanged formal PDFs.
6. **Independent gate audit** — run all tests, Python compatibility checks,
   source/input hash checks, Mutagen checks, empty-run-root check, and verify the
   one-host/one-Ala `fcntl` policy.
7. **Optimization evidence** — run the selected controls from the identical
   common initial observation for all three new seeds to exact incremental NoF
   2000.  Audit each public prefix and every attempt/result artifact.
8. **Deterministic precheck** — evaluate cost histories and correct-path guards.
   Failure ends the campaign without terminal diagnostic MD or figure changes.
9. **Fixed terminal diagnostics** — only after the precheck passes, evaluate the
   three candidate paths and paired current-IVM path at NoF 1620, 1800, and
   2000 for all ten registered diagnostic seeds (2400 NoF exactly).
10. **Final acceptance** — reproduce diagnostics from artifacts, compute the
   registered clustered simultaneous bound, and publish one write-once report
   with schema `ala2d.v17.formal_acceptance.v1`.
11. **Figures** — on pass only, render staging previews, visually inspect them,
    recheck pre-campaign PDF hashes, then require a separate explicit replace
    flag for the two formal PDFs and write complete provenance.

## Fail-closed conditions

No formal execution may begin if a placeholder remains, a seed was observed or
reused, a source/input/runtime hash differs, the Mutagen session differs, the
host-global lock is held, the run root is unexpectedly present, or the manifest
is not write-once.  The observed r5 invariant failure, or the absence of its
targeted regression under a new sealed core revision, is independently
sufficient to prohibit activation.  No acceptance or plotting may proceed
from an incomplete
`0:20:2000` public ledger, a missing/duplicate attempt, an unaccounted returned
mean force, a legacy V13 identity, one assumed paid state kind, a v1/v2 schema
mismatch, or an unpaid pending candidate.

## File inventory

- `registrations/V17_FORMAL_CAMPAIGN_DRAFT.json` — explicit non-executable
  candidate/seed/input placeholders and frozen scientific envelope.
- `PROTOCOL_DRAFT.md` — full future execution, NoF, recovery, acceptance, and
  plotting contract.
- `src/ala2d_irivm_v17.py` — V17 three-paid-state controller and deterministic
  replay.
- `src/ala2d_article_neb_reparam.py` — article centered-tangent spring map.
- `src/riemannian_torus_geometry.py` — periodic Riemannian geometry and metrics.
- `src/ala2d_oracles.py` — restrained-MD artifact/attempt implementation.
- `src/formal_contract.py` — shared identity, public-state, pending, and schema
  validators.
- `src/run_v17_formal_campaign.py` — fail-closed draft entry point, host-global
  lock, exact ledger, and NoF audit scaffold.
- `src/campaign_analysis.py` — causal initial/current-IVM/FES loaders and exact
  checkpoint scoring without LOCF.
- `src/campaign_acceptance.py` — fixed registered acceptance producer.
- `src/plot_registered_figures.py` — acceptance-gated V17 plot-data and formal
  rendering consumer.
- `src/build_causal_shared_initial.py` — shared state-zero provenance builder.
- `tests/fixtures/causal_initial_state0.npz` — 9 KiB test-only causal fixture;
  not a sealed formal input and not formal evidence.
- `tests/test_irivm_v17_core.py` — V17 state-machine/replay tests.
- `tests/test_formal_contract.py` — three-state, NoF, pending, and validation
  contract tests.
- `tests/test_run_v17_formal_campaign.py` — DRAFT gate, crash/resume, attempt
  ledger, hidden-evaluation, jobs, and lock tests.
- `tests/test_acceptance_v17.py` — V17 exact-state acceptance/schema tests.
- `tests/test_plot_registered_figures_v17.py` — V17 plot consumer and
  fail-closed replacement tests.
- `reports/R6_TEST_CONTRACT.md` — enumerated test oracle.
- `reports/V17_FORMAL_DRAFT_DESIGN.md` — implementation design notes.

The draft deliberately contains no `runs/`, copied r5 outputs, MD archive, or
generated PDF.
