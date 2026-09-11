# R6B NF30000 terminal acceptance

Accepted 2026-09-04 13:23 HKT (Asia/Hong_Kong).

## Result

- terminal status: `PASS_R6B_TERMINAL_ACCEPTANCE`;
- optimization NoF: 29,978 of the registered 30,000;
- remaining NoF: 22, insufficient for another atomic 30-internal-image band;
- canonical history records: 1,053;
- fixed-endpoint optimization NoF: 0;
- physical calls introduced by capture or verification: 0;
- locked child: clean exit, return code zero, locks released;
- primary verification SHA-256:
  `00a215f4e2953a551b7d5294780f65368b3ce4545dcd28f4623dd2645a7c1345`;
- supplemental verification SHA-256:
  `2e0516dc2a2cdeb2f72f3fcdb3b5d4b9f5ba5801e3e962bebfe2d845e141a556`.

Mutagen synchronized both verification JSON files to the local project. Their
local SHA-256 values exactly equal the remote values above.

## Accepted path versus stored charged trials

The complete history is saved, but reportability must not be confused with
budget consumption. Calls 1--818 are formally reportable. The last accepted
path is call 818 at NoF 23,512:

- discrete `I_perp = 7.248189609774711`;
- 32-image `Delta E = 104.5734104980687` kcal/mol;
- path SHA-256:
  `eb8fbfcbb5011a0467a888e32289e7cc9b8764a8f7242233ae26225239a0834f`.

Calls 819--1053, NoF 23,542--29,978, are charged trials marked
`terminal_uncommitted_charged_tail=true`. Their path, energy, and full force
arrays are all preserved and verified, but they are not accepted IR-IVM paths.
The last stored trial has `I_perp = 4.7633412812262605` and 32-image
`Delta E = 95.59051740206303` kcal/mol; these values are diagnostic only.

An independent R5B/R6B comparison checked call number, NoF, path SHA, energy
SHA, and full-force SHA for calls 1--515: 2,575 comparisons and zero mismatch.
Only the expected reportability classification changed after the former R5B
terminal tail became intermediate history in the longer run.

## Postrun invocation record

The prepared wrapper's first clean invocation exposed an argument-order bug:
it placed the runner's global `--root` option after the `verify` subcommand.
Argument parsing failed before any scientific artifact was read or any output
was written. The primary verifier was then invoked directly with the same
frozen arguments in the parser's required order and passed.

The first supplemental invocation used an incorrect relative lock-receipt
filename and failed at `Path.resolve(strict=True)` before verification or
output. The unique lock-receipt path was then taken from the frozen runner's
`LAUNCH_RECEIPT` constant; the supplemental verifier passed. With both outputs
present, the unchanged wrapper read their frozen PASS identities and emitted
`PASS_R6B_TERMINAL_ACCEPTANCE`. Neither failed invocation imported OpenMM or
made an energy or force call.
