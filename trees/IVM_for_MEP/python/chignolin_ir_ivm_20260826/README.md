# Chignolin IR-IVM transfer experiment

This directory transfers the audited Ala IR-IVM implementation to Chignolin
with the smallest necessary system changes.  The source figure and the current
Chignolin IVM baseline remain read-only under
`/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin`.

The algorithm identity is fixed as follows: a joint inexact normal proximal
subproblem, the one-way FFrbk/L-BFGS selector, selected-step multiplier
`eta=1.1`, physical atom-displacement cap `1 angstrom`, and mandatory
`state_curve` reparameterization plus its recovery transaction after every
outer iteration.  Only `alpha`, relative residual threshold `r`, and the
evaluated whole-band inner-step cap `m_max` are tuned.

The paper cost is always the sum of real joint energy/force cache misses over
all 22 images.  The migrated run limits the 20 internal images to 5000 calls;
the paper cost therefore includes two additional initial endpoint calls.  All
rejected trials and recovery calls are charged, while separate densified
barrier diagnostics are excluded from optimization cost.

See `registrations/CAMPAIGN_PREREGISTRATION.json` for the frozen search and
selection rules.  Every launched parameter setting also receives its own
pre-run JSON registration containing the complete command and source/input
hashes.

The completed experiment, including the selected setting, strict CPU1 prefix
reproduction, recovery/Number-of-Forces audit, dense-barrier comparison,
figure provenance, SHA256 inventory, and exact reconstruction commands, is
documented in `RESULTS_AND_PROVENANCE.md`.
