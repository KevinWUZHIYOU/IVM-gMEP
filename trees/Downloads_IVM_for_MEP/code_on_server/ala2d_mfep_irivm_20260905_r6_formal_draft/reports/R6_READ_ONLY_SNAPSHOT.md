# R6 V17 formal-draft read-only snapshot

Date: 2026-09-05

## Outcome

The candidate-agnostic r6 scaffolding is implemented and testable, but it is
not sealed, executable, or formal-ready.  The checked-in registration remains
exactly `DRAFT` with `formal_evidence=false`,
`new_restrained_md_authorized=false`, and `executable=false`.  It contains no
selected parameter, no formal seed, no sealed MD input, and no runtime seal.

No formal MD was started.  There is no `runs/` directory, generated PDF, or
generated PNG in r6, and no file larger than 5 MiB.  The only NPZ is the 9 KiB
synthetic/test-only common-initial fixture.

## Blocking observed failure

The r5 development screen stopped at seed 1701, trajectory ordinal 104,
incremental NoF 1940 with the invariant failure:

```text
an outer with completed movements must have anchor bounds
```

The r6 core was derived from that V17 revision.  Its core changes are limited
to formal-draft identity/schema labels and additional public-loader arrays; no
controller repair for this invariant was made.  The r5 failing run artifacts
were deliberately not copied into r6, so the failure is not reproduced by the
r6 suite.  Therefore the green synthetic tests below do not supersede the
observed failure.  Activation remains prohibited until a new controller
revision has a reproducing regression and independent verification.

## Implemented draft surface

- Candidate-agnostic DRAFT registration with explicit parameter, seed, input,
  runtime, Mutagen, host, and hash placeholders.
- V17 raw-proximal core and shared validation contract for movement, HOLD, and
  independent exact-center validation paid states.
- Exact 0:20:2000 public-prefix, pending-source-at-2000, zero-NoF refresh,
  crash/replay, attempt-ledger uniqueness, and NoF reconciliation scaffolding.
- Fail-closed runner with `jobs=1` and host-global `fcntl` lock contract.
- Acceptance and plotting consumers on the shared V17 v1 schema, with exact
  checkpoint selection and no last-observation-carried-forward.
- Candidate-agnostic protocol, activation plan, and enumerated test contract.

The chronological work plan and complete human-readable file inventory are in
`V17_FORMAL_ACTIVATION_PLAN.md`.  The scientific and state-machine design is in
`V17_FORMAL_DRAFT_DESIGN.md`.

## Verified checks

- Full r6 suite: `135 passed in 18.55s`.
- Python 3.8 `py_compile`: passed for every file under `src/*.py` and
  `tests/*.py`.
- `validate-draft`: returned `valid non-executable draft`.
- `run-optimization` against the DRAFT registration: failed before execution
  with exit code 2, as required.
- Static consumers (`campaign_analysis.py`, `campaign_acceptance.py`,
  `plot_registered_figures.py`, and `run_v17_formal_campaign.py`) contain no
  `evaluate_band(` call.
- Executable `src/` contains no hard-coded seeds 41, 42, or 43.
- Absent artifacts: `runs/`, PDFs, PNGs, and files larger than 5 MiB.

These checks validate only the non-executable draft contract; they do not
validate the failed r5 real-trajectory controller transition.

## Content inventory

`R6_FILE_SHA256.json` lists SHA-256 for all 23 authored draft files and the
test fixture, excluding generated caches, this snapshot, and the inventory
itself.  Its own SHA-256 is:

```text
f480dc928d7f0633a886b74ddb2a58bc8cc917a7a3ae7b241bcb272726fdbb63
```

The inventory was independently re-read after creation: 23 entries, zero
missing files, and zero digest mismatches.  It is a read-only draft snapshot,
not a prospective formal source seal.

## Required next gate

1. Preserve and diagnose the r5 failing prefix without altering frozen r5.
2. Implement the repair under a new controller identity/revision.
3. Add a targeted invariant/replay/NoF regression for the exact failure mode.
4. Independently rerun the full core and formal-contract suites.
5. Re-establish development evidence under the new sealed revision before
   selecting a final candidate or prospectively registering formal seeds.

Until all five steps pass, do not create a formal registration, start MD,
generate formal figures, or replace either manuscript PDF.
