# Alanine clean four-level fixed-alpha IVM audit

> **审查入口：先读 `00_READ_ME_FIRST_ALANINE.md`。** 该文件统一登记当前
> baseline、四层协议、最终整数-alpha IVM、正式论文图和归档边界；本 README
> 只保留 clean campaign 的简要摘要。

This is the clean-start Alanine (Ala2, 66-dimensional) audit frozen on
2026-08-01. Every grid member starts from the byte-identical path
`inputs/common_initial_path_equidistant.npz` (SHA-256
`cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e`).
No positions, forces, caches, quasi-Newton history, or stopping decisions are
shared between alpha values.

The completed four-level design is:

1. `stage1_exact_no_resample`: operational exact-like proximal subproblems at
   `r=0.2` and `r=0.1`, without reparameterization (20 main results).
2. `stage2_exact_resample`: the same exact-like settings with equal-arclength
   geometric reparameterization (20 main results).
3. `stage3_inexact_baseline_resample`: a plain five-inner-call inexact solve
   with reparameterization (10 main results).
4. `stage4_inexact_optimized_resample`: one uniform Phase-A34/Phase-B220
   production policy for every alpha, followed by the same reference-free,
   force-charged complete-state Pareto archive rule (10 main results and 10
   archive records).

The common alpha grid is
`0.01, 0.1, 1, 10, 20, 30, 100, 1000, 10000, inf`. The fail-closed acceptance
report selects the exact integer `alpha=20` as the only target-reaching member.
Its selected complete reparameterized state has:

- raw `I_perp = 1.8216907466e-4`;
- independently recomputed 20-subdivision barrier error
  `|Delta E - Delta E_ref| = 9.2311402e-4 kcal/mol`;
- 2,507 optimization force evaluations plus 2,301 exactly charged screening
  force equivalents, for 4,808 total;
- 17.371x lower charged cost than ODE-NEB at the matched `I_perp` threshold;
- zero rises in both metrics on the published accepted-incumbent trace.

The last statement applies to the accepted complete-state trace, not to every
trial candidate. The retained raw candidates have four densified-barrier-error
rises and a maximum error shoulder of `0.2015568 kcal/mol`; they are preserved
and disclosed in the diagnostic NPZ and acceptance report.

See `PROTOCOL.md` for the frozen method and accounting rules,
`EXPERIMENT_INDEX.md` for the code/data/figure map, and
`reports/STAGE4_V4_ACCEPTANCE_SUMMARY.md` for the acceptance scope. The formal
current machine-readable verdict is
`reports/stage4_v4_fail_closed_acceptance.json`.  The earlier
`clean_four_level_fail_closed_audit.json` remains a structural/raw-trace audit:
its raw A80 barrier-stability condition is intentionally not the accepted
v4 complete-state Pareto criterion.

Run a stage with the fixed Python environment, for example:

```bash
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  src/run_clean_ivm_grid.py --stage stage1 --jobs 4
```

The small `budget6` Stage-1 file is only a routing/accounting smoke test and is
excluded from all main-result counts.
