# 启用数据与审计清单

## 高层快照内已保存的关键文件

`final_ivm/` 保存最终整数 `alpha=20` 结论的：

- A34/B220 combined raw run；
- Stage-IV v4 reference-free Pareto archive 和它的 audit JSON；
- 最终 densified-barrier screen NPZ。

`audits/` 保存 fail-closed acceptance 结果、人类可读总结、线性计费/加速比图审计和
alpha-mechanism 标题/协议审计。

这些快照是审查副本。下面的 `python/` 路径是 plotter 实际读取的唯一可执行源。

## 图 `fig:Ala2_accuracy` 的数据

- IVM raw：`python/alanine_ivm_clean_four_level_20260801/runs/stage4_inexact_optimized_resample/stage4_uniform_v3_a20_phaseA34_phaseB220.npz`；
- IVM archive：`python/alanine_ivm_clean_four_level_20260801/runs/pareto_archive/pareto_clean_stage4_v4_a20_phaseA34_phaseB220.npz`；
- IVM dense：`python/alanine_ivm_clean_four_level_20260801/runs/diagnostics/dense_barrier_clean_stage4_pareto_screen_a20_phaseA34_phaseB220.npz`；
- ODE--NEB raw/metrics/dense：`python/alanine_v10_chignolin_v9_fixed_alpha_20260730/formal_figure_inputs/ala_0to1_N22_k50_alpha0.0001_ode_results.npz`、
  `formal_figure_inputs/metrics_ala_0to1_N22_k50_alpha0.0001_ode_results.npz`、
  `runs/diagnostics/dense_barrier_ala_ode_neb_k50_logsample_dense.npz`；
- Ramachandran surface：`python/alanine_v10_chignolin_v9_fixed_alpha_20260730/formal_figure_inputs/ala2_scan_cache.pkl`。

## 图 `fig:Ala2_loss_history` 的数据

IVM 使用上述 Stage-IV v4 数据；ODE--NEB 使用上述 formal inputs。其他长预算 baseline 是：

- Original String：`python/alanine_v10_chignolin_v9_fixed_alpha_20260730/runs/publication_long90k_20260802/publication_long90k_20260802_publication90k20260802r2_original_nolambda_dt0p014_r1_cfg587b411dd523_budget4500.npz`；
- Static NEB：`python/alanine_v10_chignolin_v9_fixed_alpha_20260730/runs/neb/neb_publication90k20260802r2_central_k30_alpha0p013_static_budget4500.npz`；
- OBSM--BFGS：`python/alanine_v10_chignolin_v9_fixed_alpha_20260730/runs/publication_long90k_20260802/publication_long90k_20260802_publication90k_robust_20260802r2_bfgs_lam0p3_inner1_r1_nocap_cfg9c7658b737c3_budget8000.npz`；
- OBSM--FIRE：`python/alanine_v10_chignolin_v9_fixed_alpha_20260730/runs/publication_long90k_20260802/publication_long90k_20260802_publication90k20260802r3_fire_lam0p85_inner3_r1_nocap_cfg75c2b3203822_budget4510.npz`；
- densified-barrier histories：同一 baseline 谱系下的
  `runs/diagnostics/dense_barrier_ala_original_dt0p014_publication90k.npz`、
  `runs/diagnostics/dense_barrier_ala_static_k30_dt0p013_publication90k.npz`、
  `runs/diagnostics/dense_barrier_ala_obsm_bfgs_lam0p3_robust_publication90k.npz` 和
  `runs/diagnostics/dense_barrier_ala_obsm_fire_lam0p85_inner3_dt0p1_publication90k.npz`。

## 图 `fig:Ala2_alpha_mechanism` 的数据

只使用 `alpha = 0.01, 0.1, 1, 10, 100, 1000, 10000`。

- Stage I：`python/alanine_ivm_clean_four_level_20260801/runs/stage1_exact_no_resample/`
  中 `r0p1` 的对应 decade-alpha 长预算 run；实际被 loader 选中的 budget
  为 `1000,1000,2500,1000,1000,1000,1000`；
- Stage IV：`runs/stage4_inexact_optimized_resample/stage4_uniform_v3_a*_phaseA34_phaseB{600,1000}_fullbudget.npz`，
  其中 `alpha=1,10` 使用 B1000，其余使用 B600；
- 计费 archive：`runs/pareto_archive/pareto_clean_stage4_v7_min5000_a*_phaseA34_phaseB{600,1000}.npz`；
- plotter 以每个 combined NPZ 中登记的 `phase_a_result`/`phase_b_result` 路径回读两阶段元数据，
  并由 `TITLE_PROTOCOL_AUDIT.json` 验证 `r=0.1` 与 `r=0.96 -> 0.8`。

所有大体积 raw 数据都不在 `paper/` 中复制，避免出现多个同名“真实源”。
本 bundle 内容的完整性由根目录 `SHA256SUMS` 固定。
