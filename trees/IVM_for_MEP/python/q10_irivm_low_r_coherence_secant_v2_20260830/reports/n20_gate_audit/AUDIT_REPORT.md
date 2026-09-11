# QM10 low-r coherence-conditioned IR-IVM result audit

- Campaign: `q10-irivm-low-r-coherence-secant-v2-20260830`
- Audit status: **complete**
- Registered runs requested: 4
- Integrity-valid results: 4
- Integrity failures: 0
- Strict rule: at every one of 51 latter-half same-NoF LOCF checkpoints, both $I_\perp$ and $|\Delta E-\Delta E_{\rm ref}|$ must be exactly lower than frozen Article IVM; equality fails.
- Recovery audit: every complete projected candidate's path, energies, and gradients, the strong/conservative/ordinary branch, stationarity, curvature-certified and curvature-rejected sets, strong-branch rank omissions, published indices, normalized curvatures, and exact tie break are independently recomputed. Optimizer publication helpers and reported scores are not trusted.
- Reference densified barrier: `67.50188060819488`

## Outcome

- $N_{\rm int}=20$: 2 strict passing run(s).
  Best available: `alpha=1e-3, r=0.65, m_max=4` (`c14cc0cf2a7f6171`), 51/51 joint checkpoint wins, worst joint ratio 0.990748.

## Recovery secant publication totals

- Curvature-certified: `601`; published: `98`; curvature-rejected: `219`; rank-omitted: `503`.
- Accepted recovery branches: strong top-1 `37`, conservative all `1`, ordinary all `3`.

## Integrity

All requested registered results, receipts, hashes, finite-value checks, and Number-of-Forces ledgers passed.

A lack of a strict winner is reported as a scientific non-pass, not as a program error. Only integrity failures make this audit exit non-zero.
