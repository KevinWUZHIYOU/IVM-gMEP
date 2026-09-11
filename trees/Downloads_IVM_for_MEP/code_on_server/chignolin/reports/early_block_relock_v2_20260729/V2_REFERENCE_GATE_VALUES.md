# Frozen cap=15 reference values for unified V2 gates

These values were extracted before any complete unified V2 result file
existed. They instantiate, without changing, the metrics defined in
`V2_SCORING_PREREGISTRATION.md`.

## Preserved reference run

- Result:
  `ivm_alpha_mechanism_proximal_only_a1_full_central_state_ivm_cfg85ee8e3b477d_budget500.npz`
- Final total Number of Forces: 6823
- Final unnormalized `I_perp`: 0.03885242073674519
- Final equal-speed integral: 5.885353699911007e-20
- Frozen adaptively densified barrier reference:
  20.373739515742795 kcal/mol

## Unnormalized `I_perp`

| Requested total-force budget | Actual endpoint | Endpoint | Last-30% arithmetic mean |
|---:|---:|---:|---:|
| 1000 | 982 | 0.33320328373410885 | 0.42587241402206116 |
| 3000 | 2982 | 0.09780896579668730 | 0.11532450448540735 |
| 6000 | 5982 | 0.04638147925468567 | 0.05396358532234828 |
| 10000 | 6823 | 0.03885242073674519 | 0.04698488859932258 |

## Adaptively densified barrier error

Every row uses seven real OpenMM paths in the final 30% window and the
10→20→40/80 adaptive segment protocol.

| Requested total-force budget | Mean absolute error (kcal/mol) | RMS error (kcal/mol) | Temporal standard deviation (kcal/mol) |
|---:|---:|---:|---:|
| 1000 | 5.081738216756542 | 5.102191992347510 | 0.456398754524032 |
| 3000 | 2.710627100271327 | 2.719024557285779 | 0.213530481191291 |
| 6000 | 0.582281975044548 | 0.717216882200115 | 0.421414438705414 |
| 10000 | 0.561691639098925 | 0.798519886390451 | 0.574402494925906 |

The production non-inferiority gate compares unified V2 to these same
statistics, not to a zero-error single endpoint.
