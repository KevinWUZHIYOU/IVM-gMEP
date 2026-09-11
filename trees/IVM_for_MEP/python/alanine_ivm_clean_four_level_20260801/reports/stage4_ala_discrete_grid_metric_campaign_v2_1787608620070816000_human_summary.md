# Ala discrete-grid metric campaign v2: audited human summary

Status: complete. This document summarizes the frozen machine-readable analysis; it does not replace it. Repository/study root:

`/Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801`

If this prose conflicts with the registered JSON artifacts, the JSON content identified by the SHA-256 values below is authoritative.

## Scope and accounting

- Optimizer grid: 240 `inexact + reparam` cells, with
  `alpha in {1e-1,1e0,...,1e6}`, `r in {0.1,0.3,0.5,0.7,0.8}`,
  `inner_max_steps in {5,8,10,15,20,50}`, and a hard total internal
  optimization-NF budget of 5000 per cell.
- Secondary grid: 8 `exact + no-reparam` alpha cells. Exact-v2 completed
  after the failed v1 wrapper attempts were preserved separately; the two v1
  attempts each spent 20 initial-band optimization NF and published no result.
- Formal state eligibility:
  - inexact: initial or reparameterization-committed, complete, and mesh
    `CV <= 0.0065`, `max/min <= 1.03`;
  - exact: registered exact-complete mask and the same mesh gate;
  - terminal charged incomplete tails are excluded.
- Raw `I_perp` is read from each frozen optimizer result. It is not smoothed,
  interpolated, or replaced by a terminal value.
- Densified barrier for every eligible unique path is evaluated at all four
  nested Cartesian levels `20,40,80,160`; level 160 (3361 physical points for
  a 21-segment path) is selected. The metric is
  `Delta E = (max(E_dense) - E_start) * 23.0605 kcal/mol`.
- Dense convergence requires `abs(B160-B80) <= 0.001 kcal/mol`. All 1403
  unique paths passed. The two-host common-initial sentinels agreed exactly at
  all four levels.
- The 5916 eligible state references deduplicated to 1403 unique paths. The
  dense diagnostic used 4,715,483 unique-path force evaluations plus 6,722
  sentinel evaluations, exactly 4,722,205. These diagnostic evaluations are
  kept separate from, and never added to, optimizer NF.

## Registered formal conclusion

The registered targets are raw `I_perp <= 2.1e-4` and densified barrier error
`abs(Delta E - 2.4560825131253083) <= 0.017 kcal/mol`, at an actually observed
eligible state, with no interpolation.

- 23/248 grid cells reached the `I_perp` target.
- 94/248 grid cells reached the barrier-error target.
- 0/248 reached both targets at the same eligible recorded state.
- The frozen incumbent reached the individual targets at NF 2150 and NF 911,
  respectively, but also did not reach them jointly.
- Therefore 0/249 candidates passed the registered joint winner gate and the
  formal result is `no_unique_winner`.
- The grid-only full-curve nondominated set has 162 cells. Adding the incumbent
  gives 163. The incumbent full-curve dominates 50 grid cells; the remaining
  198 are incomparable with it. No grid cell full-curve dominates the
  incumbent.

This is not a tie and does not mean the incumbent is uniformly best. The
curves cross, and the registered partial order intentionally refuses to turn
those crossings into a scalar ranking.

## Engineering recommendation (not the formal winner rule)

For a single follow-up center, use:

`inexact + reparam, alpha=10, r=0.5, inner_max_steps=8, total NF budget=5000`

Cell ID: `c6b759e30de39e91`. Frozen result SHA-256:
`5e2911ea5f85af4214c608cf704fc2ac7ba03c4b1fbe7acd62ca085e35e34b4a`.

Rationale:

1. As an explicitly post-hoc engineering diagnostic, minimize over observed
   eligible states
   `max(I_perp/2.1e-4, barrier_error/0.017)`. This cell is one of the five
   bitwise-identical cap-8-or-larger co-minima among all 248 grid cells plus
   the incumbent: `1.8497158092` at NF 2689,
   with `I_perp=1.71022548174e-4`, `Delta E=2.48752768188 kcal/mol`, and
   barrier error `0.0314451687569 kcal/mol`. It passes the `I_perp` target but
   still misses the barrier-error target; the diagnostic must not be called a
   formal winner.
2. Its `I_perp` first passage is NF 2276. Its final eligible record is NF 3569,
   `I_perp=1.44750919287e-4`, barrier error `0.0462557838613 kcal/mol`.
3. At `alpha=10,r=0.5`, caps 8, 10, 15, 20, and 50 give bitwise-identical
   scientific trajectories and dense metrics. Cap 8 is the smallest tested
   nonbinding value: 0 cap stops, all 23 fixed-path solves stopped by the
   residual criterion, and the largest zero-based inner step index was 5.
4. Cap 5 binds once and first diverges at stored call 13. It reaches the
   `I_perp` target only 11 NF earlier (2265), but its final barrier error is
   substantially worse (`0.0965246791779` versus `0.0462557838613`). Its
   full curves are incomparable with cap 8, so cap 5 is not a lossless
   reduction.

Alternative objective-specific cells:

| Objective | Controls | Observed result |
|---|---|---|
| Fastest grid `I_perp` target | `alpha=10,r=0.5,c=5` | NF 2265; one cap stop |
| Deepest grid `I_perp` | `alpha=10,r=0.1,c=8` | `1.16501647282e-4` at NF 3834; barrier error `0.117761971089` |
| Fastest barrier-error first passage | `alpha=10,r=0.7,c=5` (all tested caps tie here) | NF 672; `I_perp` target at NF 3386, never jointly |
| Best balanced same-state proximity | `alpha=10,r=0.5,c=8` | NF 2689; relative worst-target factor `1.84972` |

## Controlled-variable patterns

### Alpha

Descriptive main effects across 30 matched `(r,cap)` contexts per alpha:

| alpha | cells reaching `I_perp` target | cells reaching barrier target | interpretation |
|---:|---:|---:|---|
| 0.1 | 0/30 | 0/30 | too strongly damped within 5000 NF |
| 1 | 0/30 | 30/30 | barrier first-passage specialization; the barrier later rebounds |
| 10 | 23/30 | 12/30 | clear `I_perp` sweet spot and best dual-metric compromise |
| 100 | 0/30 | 7/30 | cap-sensitive; no `I_perp` success |
| 1e3 | 0/30 | 9/30 | cap-sensitive; no `I_perp` success |
| 1e4 | 0/30 | 11/30 | cap-sensitive; no `I_perp` success |
| 1e5 | 0/30 | 13/30 | cap-sensitive; no `I_perp` success |
| 1e6 | 0/30 | 12/30 | cap-sensitive; no `I_perp` success |

Thus increasing alpha is not monotone. `alpha=10` is a sharp regime change:
lower alpha under-progresses in `I_perp`, while higher alpha spends many inner
steps against the cap and loses outer/reparameterization opportunities.

### Residual ratio r

Across 48 matched `(alpha,cap)` contexts per level, `r=0.5` gives the fastest
`I_perp` first-passage distribution (minimum NF 2265, median 2276 among its six
reaching cells) and the best same-state compromise. `r=0.7` has the most
barrier first-passage successes (30/48) and the fastest barrier passage
(NF 672), but the barrier generally rebounds before `I_perp` becomes tight.
`r=0.8` has no grid cell reaching the `I_perp` target. These are interactions
with alpha and cap, not context-free single-factor laws.

### Inner max steps

154/240 inexact cells bind their cap at least once, for 1853 cap stops. Binding
is absent for alpha 0.1 and 1, appears in 4/30 alpha-10 cells, and occurs in
all 30 cells at every alpha >=100. Even cap 50 binds in 25/40 contexts (90
stops), so it must not be described as operationally infinite. Increasing the
cap is therefore not monotone and can reduce the number of outer or
reparameterization opportunities available within the same total NF budget.

### Exact + no-reparameterization package

None of the eight exact/no-reparameterization cells reaches either registered
metric target after the formal mesh gate. Mesh eligibility is the immediate
problem: alpha 0.1 has 250 exact-complete states but only 4 eligible states;
alpha 1 has 121/1; alpha 10 has 49/1; alpha 100 has 10/1; alpha >=1e3 has 3/1.
In the closest frozen same-alpha comparisons, alpha 0.1 is incomparable and
the inexact/reparameterized package full-curve dominates for alpha 1 through
1e6.

This comparison jointly changes exactness, reparameterization, and the
operational cap. It supports the operational conclusion that the tested exact
package fails to maintain a usable mesh; it does not identify
reparameterization alone as the causal factor.

## Frozen provenance

| Artifact | SHA-256 |
|---|---|
| metric-v2 preregistration | `5fcfad3228e72c267ea1d768dd6f677528ae1fc8e8c6f180ea0d3498503b98a3` |
| local host report | `59c55a351f9babe66ed4a91bf16e94c4291e0883a5fe6f2213982b915e0b9f83` |
| remote host report | `0c18528c97224ea458d717f34d9d4e113cf1ceff7b18cf7c6b413d868f21c248` |
| metric aggregate | `0e53a4e00a8a30f8e60c9fd278426f9b55fecada08840dee6f514a4e93390e63` |
| formal metric analysis | `e6724142f780aefb567d060abfca8843b47fd27bf6cbc3bda34ea6156d194b3a` |
| metric campaign source | `4e05ec7896d993d214b121ae9ce9711121a6058407637d50a926e391a36b97f1` |
| analysis source | `b84986307bcb8251afa42a4ba046a3b669c5e2395bbbfeed134ec792d42c7b11` |
| portable decision snapshot | `5da036ff5f8959c6e5cd8398ab3cedc7746f862deaf713b89edc669725d6da10` |
| decision manifest | `38ec86537581240aadc55bdae4a25d1e4e14b9c87d5c432eda4b74a6edb2a00a` |
| formal barrier reference audit | `bbb8486e222b3187caccc6831917464d72f42588fe949fc9c49c6185b81f901d` |
| optimizer successor preregistration | `d3ebf3239931dfde104342b791d8ed096b4ddef597c1bee0de4cb32b4b289aa3` |
| inexact optimizer aggregate | `95b2be41434f452a58c49698af89b2ccf993c933e12667476af442da84250807` |
| exact optimizer aggregate | `a2307bd55d7b165d07e3d83c61c6c5b41f5449b7db72d66f540ffaf74cf0556e` |
| incumbent result | `1c81ad6fe876a0eb94f2fc55d5d2567737face95ff52b6fe1bdfffc536d16787` |
| superseded-v1 zero-launch ledger | `bdd2f5ce3117faa7446aebf89c51b482a54a59d3b55bf6909efefc27fbf47a5e` |

The formal analysis JSON contains the complete cell summaries, observed CTP
frontiers, pairwise partial-order results, matched-factor effects, exact versus
inexact package comparisons, and the SHA map for all 1403 diagnostic artifacts.
