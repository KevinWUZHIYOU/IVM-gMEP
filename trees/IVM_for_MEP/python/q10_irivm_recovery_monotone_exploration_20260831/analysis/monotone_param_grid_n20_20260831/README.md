# N20 exact-envelope parameter search

This is an exploratory, non-preregistered parameter search.  It keeps the
`exact_envelope` optimizer source unchanged and evaluates one run at a time
with `jobs=1`, a total force-call budget of 3000, HJ improved-tangent spring
reparameterization, integer-scientific-mantissa `alpha`, `r < 0.7`, and
`m_max >= 4`.

Numerical environment:

- `OPENBLAS_NUM_THREADS=8`
- `PYTHONHASHSEED=0`
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and
  `NUMEXPR_NUM_THREADS` unset

Optimizer SHA-256:
`af813409f2b2293d86891f6e90c00a2cb294c4fa50e52fe6b3bb5cd78fefd44a`.

## Result

Among 28 independently scored N20 results, the only result satisfying all 98
Article-IVM checkpoints for both `I_perp` and the densified barrier error is
still:

| alpha | r | m_max | I rises | max rise | strict I | strict barrier | strict joint | terminal NoF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-3 | 0.3 | 8 | 1 | 1.0385322626e4 | 98/98 | 98/98 | 98/98 | 2363 |

The sole rise is the trusted provisional publication at NoF 80:

- before, NoF 60: `I_perp = 14975.056988037288`
- after, NoF 80: `I_perp = 25360.379614046906`
- rise: `10385.322626009618`
- the same step reduces the densified barrier from `108.87907023019142` to
  `101.4693777615358`; the next public state at NoF 100 reduces `I_perp` to
  `10528.695713273204`.

For `alpha=1e-3`, changing `r` over `{0.1,0.2,0.3,0.4,0.5,0.6}` at
`m_max=8`, or changing `m_max` over `{4,5,6,7,8,10}` at `r=0.3`, never
removes the NoF-80 rise.  The rise has exactly the same before/after values
in every such run.  The best alternatives reach only 93/98 joint strict
checkpoints.

Additional screened controls:

| screened controls | completed results | best strict joint | rise-count range |
|---|---:|---:|---:|
| alpha=5e-4, r in 0.1--0.3, m_max in {4,5,6,7,8,10} | 15 | 0/98 | 1--3 |
| alpha=2e-3, r=0.3, m_max=8 | 1 | 1/98 | 1 |
| alpha=3e-3, r=0.3, m_max=8 | 1 | 1/98 | 3 |

Three attempted `alpha=5e-4, m_max=6` controls (`r=0.1,0.2,0.3`) stopped
without a scored result at the fail-closed invariant
`main entered with an open recovery provisional`.  A later queued run was
interrupted before result publication when the search was explicitly stopped;
it is not counted above.

## Interpretation

Within the completed parameter cross-sections, `r` and `m_max` cannot remove
the remaining rise because it occurs in the early trusted provisional branch
before those recovery controls distinguish the trajectories.  Changing
`alpha` does alter the trusted trajectory, but every tested alternative both
retains at least one rise and loses the 98/98 scientific acceptance result.
Therefore parameter tuning alone did not produce zero rises under the retained
trusted mechanism.  Removing the last rise requires a resolution-independent
recovery/publication rule change, followed by fresh preregistration and the
full independent audit; it cannot honestly be claimed from this grid.

Raw immutable-per-run outputs are under
`../../runs/exact_envelope_n20_*/summary.json` and the matching
`trajectory.npz` files.  The runner uses `exist_ok=False`, so no prior result
was overwritten.
