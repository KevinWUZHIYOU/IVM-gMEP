# Alanine 66D 实验总说明：审查前必须先读

状态：正式实验谱系，科学结果冻结于 2026-08-02；高层登记表更新于 2026-08-04。

这份文件是 Alanine（Ala2，66 个笛卡尔自由度）算例的最高优先级说明。
本文件就是论文侧的高层审查入口。它与
`/Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801/00_READ_ME_FIRST_ALANINE.md`
及同目录的 `00_READ_ME_FIRST_ALANINE_副本.md` 保持同步。
审查、复现实验或继续调参前，必须先读这份总说明；不要从目录名中的 `v49`、
`v67`、`v150`、`20.8` 或某张旧图反推当前论文结论。

当前正式结论只由以下四个入口共同构成：

1. 论文侧的 Ala 启用版本汇总：
   `/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/alanine_66d/`；
2. 四层 clean-start IVM 的可执行源码与原始数据：
   `/Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801/`；
3. 独立调优的 baseline 可执行源码与冻结结果：
   `/Users/wuzhiyou/IVM_for_MEP/python/alanine_v10_chignolin_v9_fixed_alpha_20260730/`；
4. `paper/main.tex` 当前实际引用的稳定 PDF：
   `/Users/wuzhiyou/IVM_for_MEP/paper/Figures/ala2_66d/official_20260802/`。

旧 V49--V67 continuation、非整数 `alpha=20.8` 和旧的
`15.2x/16.7x` 论文数字已退出正式谱系。旧文件保留在归档子目录中，只用于追溯，
不得重新放回 `main.tex`。

## 0. 高层文件层级和当前启用范围

- 论文源唯一真值仍是 `/Users/wuzhiyou/IVM_for_MEP/paper/main.tex`。
  判断“论文采用了哪张图”时，以它的 `\\includegraphics` 为准，
  不以任何 candidate 目录的时间戳为准。
- `paper/ActiveExperiments/alanine_66d/` 是新的高层审查入口；
  `figures/` 中的三个 PDF 与 `main.tex` 引用的三个 official PDF
  逐字节相同，`code/` 是启用代码的冻结审阅快照，
  `data/` 保存最终 IVM 关键数据、机器审计和全部大数据的唯一源路径。
- 高层 `code/` 快照主要用于审查和防止版本混淆。重跑时仍应按本文件给出的
  `python/.../src/` 源路径执行，因为最终 plotter 使用了基于仓库根目录的数据定位。
- `paper/ExperimentArchive/` 和
  `paper/Figures/ala2_66d/archive_legacy_pre_clean_20260802/` 都是过时区；
  `paper/Figures/ala2_66d/clean_*` 是中间 candidate/build 输出。它们可用于追溯，
  但不是当前论文资产。

## 1. 共同物理问题、初值和计费

- 体系：真空 Alanine dipeptide，Amber14/OpenMM，22 个原子，66D。
- 路径：22 个 images，其中 20 个为可动内部 images。
- 拓扑：`inputs/ala2_fixed.pdb`，SHA-256
  `7ffaf924e345b6fbbba06d0388a67d18f04c07729a2ab7d65ede9a61cd27eda2`。
- 共同初始路径：`inputs/common_initial_path_equidistant.npz`；其中笛卡尔数组
  SHA-256 为
  `cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e`。
- 所有方法使用 normalized central-difference tangent
  `x[i+1] - x[i-1]`。所有 alpha 成员从干净进程开始；位置、力缓存、L-BFGS
  历史和停机决策不跨 alpha 共享。
- 每次 OpenMM cache miss 记为一次 joint energy--force query。被拒绝的 line
  search、未完成的 inner solve、continuation 初始化、reparameterization recovery
  均计费。图中 `N_F` 使用保存的全路径计数；Pareto densified screen 的真实
  OpenMM 调用也作为 force-equivalent 加入。
- `I_perp` 使用 raw 值；不把 cumulative minimum 冒充原始轨迹。
- 稀疏 image barrier 只用于运行时诊断。正式 barrier 使用 densified path 独立重算。
- 正式参考 barrier 为
  `Delta E* = 2.4560825131 kcal/mol`，但 Stage-IV Pareto 选择过程不读取该值。

## 2. Baseline：算法、调参和最终配置

所有 baseline 只在同一种方法内部排序，不能利用 IVM 或其他方法的结果挑参数。
粗筛使用真实成本下的 `I_perp`、raw barrier 和 mesh quality；finalist 再做独立、
计费的 densified-barrier 诊断。总调度代码为：

- `src/run_competitor_screen.py`；
- `src/analyze_competitor_screen.py`；
- 详细旧审计：`COMPETITOR_BENCHMARKS.md`；
- dense barrier：`src/postprocess_dense_barrier.py`。

以上相对路径在
`/Users/wuzhiyou/IVM_for_MEP/python/alanine_v10_chignolin_v9_fixed_alpha_20260730/`
下。

### 2.1 Original String

算法：每个 outer step 对所有内部 images 做一次 full physical-force Euler 更新
`x <- x + dt F`，随后每一步都做等弧长重参数化。这里不是调用 SciPy 的默认
spline；`run_string_family.py` 直接复用 `run_obsm.py` 中按 Simplified String
Method Appendix A 实现的 cubic spline：先解文中 `sigma` 的对称三对角系统，
两端三阶导数由首尾四点插值 cubic 给出，再在 uniform chord-length labels 上取值。

调参：

- coarse `dt`：`0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006,
  0.008, 0.010, 0.0125, 0.015, 0.020, 0.030, 0.040`；
- fine `dt`：`0.0105, 0.011, 0.0115, 0.012, 0.01225, 0.0125,
  0.01275, 0.013, 0.0135, 0.014`；
- long finalists：`0.013, 0.0135, 0.014`；
- reparameterization interval 固定为 1，`string_length_tol=0`，因此不会用
  string-length change 提前停机。

最终：`dt=0.014`，3000 calls，`N_F=60002`，
`I_perp=3.265520e-3`，densified barrier `2.4737948 kcal/mol`，
误差 `0.0177123 kcal/mol`。

代码和数据：

- runner：`src/run_string_family.py`；
- 最终 raw 数据：
  `runs/competitor_screen_original_finalists/competitor_screen_original_finalists_finalists20260731_original_nolambda_dt0p014_r1_cfg587b411dd523_budget3000.npz`；
- dense 数据：`runs/diagnostics/dense_barrier_ala_original_dt0p014_finalist.npz`；
- 排名：`reports/competitor_screen_original_finalists_budget3000_ranking.json`。

### 2.2 Static NEB

算法：ASE `NEB` 数据结构和 ASE `NEBOptimizer(method="static")`；`alpha=dt`
是传给 ASE static optimizer 的显式 Euler 步长，`k` 是 spring constant。

调参：

- coarse `k`：`0.1, 1, 5, 10, 20, 40, 50, 80, 100`；
- coarse `dt`：`0.001, 0.002, 0.004, 0.006, 0.008, 0.010, 0.0125,
  0.016, 0.020`；
- fine `k`：`20, 30, 40, 50, 60`；
- fine `dt`：`0.012, 0.01225, 0.0125, 0.01275, 0.013`；
- long finalists：`(k,dt)=(20,0.013),(30,0.0125),(30,0.01275),
  (30,0.013),(40,0.0125),(50,0.0125)`。

最终：`k=30, dt=0.013`，3000 calls，`N_F=60002`，
`I_perp=3.761558e-3`，densified barrier `2.4731931 kcal/mol`，
误差 `0.0171106 kcal/mol`。

代码和数据：

- runner：`src/run_neb_competitor.py --optimizer static`；
- 最终 raw 数据：
  `runs/neb/neb_finalists20260731_central_k30_alpha0p013_static_budget3000.npz`；
- dense 数据：`runs/diagnostics/dense_barrier_ala_static_k30_dt0p013_finalist.npz`；
- 排名：`reports/competitor_screen_static_finalists_budget3000_ranking.json`。

### 2.3 ODE--NEB，以及 ASE 实现边界

必须准确区分以下事实：

- 实际参数名是 `optimizer="ode"`，不是 `optimizer="pde"`；代码中没有
  `pde` 这个合法选项。
- ODE 和 static 都由 ASE 的 `NEBOptimizer` 积分；我们没有另写一套 ODE/static
  NEB optimizer。
- 但它也不是“完全未修改的原生 ASE tangent”。`ala_core.py` 中的
  `CentralDifferenceNEB(ase.mep.neb.NEB)` 与
  `CentralDifferenceNEBMethod(ase.mep.neb.NEBMethod)` 是很薄的扩展，只把
  ASE NEB method hook 的 tangent/projection 换成所有方法共用的 normalized
  central difference；ASE 仍负责 NEB state/force plumbing 和 optimizer。
  因此最准确表述是“ASE NEB/NEBOptimizer + audited central-tangent method
  extension”，不能写成自研完整 NEB，也不能写成完全未改动的 ASE `aseneb`。

ODE spring scan 的可追溯 coarse grid 为
`k=0.1,1,5,10,20,50,80,100,500,1000,2000`，接口 `alpha=1e-4`；ASE ODE
update 本身不使用该 `alpha`。早期 500/2000-call raw-barrier composite screen
曾把 `k=1` 排在第一，但它不是当前正式高精度曲线；`k=1` 的 5000-call
`I_perp=6.9941e-4`。最终全方法图冻结的是已有的高精度正式输入 `k=50`，其
5001-call `I_perp=2.040680e-4`。因此不能写“k=50 是短程 k-screen 的 rank 1”；
应写“正式高精度 ODE 输入选用 k=50”。

最终：`k=50`，terminal call 5001，`N_F=100022`，
`I_perp=2.040680e-4`，densified barrier `2.4921556 kcal/mol`，
误差 `0.0360731 kcal/mol`。

代码和数据：

- 可重跑 runner：`src/run_neb_competitor.py --optimizer ode --spring-k 50`；
- 正式冻结 raw input：
  `formal_figure_inputs/ala_0to1_N22_k50_alpha0.0001_ode_results.npz`；
- metric input：
  `formal_figure_inputs/metrics_ala_0to1_N22_k50_alpha0.0001_ode_results.npz`；
- dense 数据：`runs/diagnostics/dense_barrier_ala_ode_neb_k50_logsample_dense.npz`；
- coarse scan 代码/报告在相邻旧研究
  `python/alanine_production_alpha_20260730/src/{run_grid.py,analyze_neb_scan.py,
  analyze_neb_extensions.py}` 和 `reports/neb_*`。

### 2.4 OBSM--BFGS

算法严格区分于 IVM：在当前 string 的 normalized central tangent 所定义的冻结
normal hyperplane 上，逐 image 最小化裸势能 `U`，不含 IVM proximal 项；再以
`x_new=(1-lambda)x_old+lambda*x_min` 混合，并按设定频率使用上面的 Appendix-A
cubic spline 重参数化。inner solver 是 OBSM Section 3 的 full dense BFGS，
采用 paper-`p` strong-Wolfe line search；被拒绝的试探都计费。

主要调参范围：

- broad：`lambda=0.10,0.25,0.50,0.75,0.90`，inner steps
  `1,3,5,8`，reparameterization interval `1,3,5,10`；
- identity-H0 stage：`lambda=0.25,0.50,0.75,0.90`，inner `1,3,5`，
  interval `1,5`；
- bonded-curvature stage：`lambda=0.02,0.05,0.10,0.20`，inner `1,3`，
  interval `1,5`；
- small identity stage：`lambda=0.002,0.005,0.01,0.02`，inner `2,3`，
  interval `1,5`；
- mid identity：`lambda=0.05,0.10,0.15,0.20`，inner `1,2,3`，interval 1；
- cross-outer carry：`lambda=0.10,0.20,0.30`，inner 1，interval 1；
- finalists：`lambda=0.20,0.25,0.30`，并做 interval 5 sensitivity。

最终：`lambda=0.3`，每 image 每 outer 一次 inner BFGS step，interval 1，
bonded-force-field `H0`，跨 outer curvature carry；选 balanced checkpoint call 91，
`N_F=1579`，`I_perp=0.4501776`，densified barrier `2.4886676`，
误差 `0.0325851 kcal/mol`。完整 300-call raw history 保留并显示后续 oscillation，
没有用 running-best envelope 隐藏。

重要披露：bonded `H0` 是只依赖坐标/力场项、对共同初始路径预计算且不调用
OpenMM 的 preconditioner；跨 outer carry 也是实现选择。这个 finalist 不能冒充
“严格 identity-H0、每轮清空”的原论文基线。

代码和数据：

- runner：`src/run_obsm.py --solver bfgs`；
- preconditioner：`src/precompute_bfgs_bonded_h0.py`、
  `src/forcefield_preconditioner.py`；
- cache：`inputs/obsm_bfgs_common_initial_bonded_h0.npz`；
- 最终 raw 数据：
  `runs/competitor_screen_bfgs_diagnostic_finalists/competitor_screen_bfgs_diagnostic_finalists_diagnostic_finalists20260731_bfgs_lam0p3_inner1_r1_nocap_cfg3958c11e3826_budget300.npz`；
- dense 数据：`runs/diagnostics/dense_barrier_ala_obsm_bfgs_lam0p3_r1_finalist.npz`；
- 排名：`reports/competitor_screen_bfgs_diagnostic_finalists_budget300_ranking.json`。

### 2.5 OBSM--FIRE

外层 OBSM 与 BFGS 版本相同；inner solver 改为 Section-3 FIRE 的显式 Euler
实现：full physical force 更新 velocity，只有 `dx/dt` 投影到冻结 hyperplane。
重参数化仍是同一 Appendix-A cubic spline。

调参：

- broad：`lambda=0.25,0.50,0.75,0.90`，inner `1,3,5`，interval `1,5`，
  `dt=0.02,0.05,0.10`；更宽的通用网格还覆盖 `lambda=0.10`、inner 8、
  interval `3,10` 和 `dt=0.01`；
- refine：`lambda=0.70,0.75,0.80,0.85`，inner `2,3,4`，interval 1，
  `dt=0.08,0.10,0.12,0.15`；
- finalists：`(lambda,inner,dt)=(0.75,3,0.10),(0.80,3,0.10),
  (0.85,3,0.10),(0.75,2,0.12),(0.80,2,0.12),(0.85,2,0.12)`；
- mass 固定 1，beta 固定 0.1。

最终：`lambda=0.85`，inner 3，interval 1，`dt=0.1`，terminal call 1999，
`N_F=39982`，`I_perp=7.086426e-3`，densified barrier `2.5147448`，
误差 `0.0586623 kcal/mol`。

代码和数据：

- runner：`src/run_obsm.py --solver fire`；
- 最终 raw 数据：
  `runs/competitor_screen_fire_finalists/competitor_screen_fire_finalists_finalists20260731_fire_lam0p85_inner3_r1_nocap_cfg3416aa96d5bf_budget2000.npz`；
- dense 数据：`runs/diagnostics/dense_barrier_ala_obsm_fire_lam0p85_inner3_dt0p1_finalist.npz`；
- 排名：`reports/competitor_screen_fire_finalists_budget2000_ranking.json`。

### 2.6 正式图使用的长预算真实性验证

为避免短轨迹在 90k 线性横轴上被视觉外推，正式两指标图重新从共同初始路径
运行了所有非 IVM baseline。左图直接读取完整 raw counter ledger，不做水平延长，
每条非 IVM 轨迹都真实达到至少 `N_F=90002`：

- Original String：call 4500，`N_F=90002`，terminal
  `I_perp=1.437651e-3`；raw
  `runs/publication_long90k_20260802/publication_long90k_20260802_publication90k20260802r2_original_nolambda_dt0p014_r1_cfg587b411dd523_budget4500.npz`；
- Static NEB：call 4500，`N_F=90002`，terminal
  `I_perp=1.688783e-3`；raw
  `runs/neb/neb_publication90k20260802r2_central_k30_alpha0p013_static_budget4500.npz`；
- OBSM--FIRE：call 4509，`N_F=90182`，terminal
  `I_perp=1.728164e-3`；raw
  `runs/publication_long90k_20260802/publication_long90k_20260802_publication90k20260802r3_fire_lam0p85_inner3_r1_nocap_cfg75c2b3203822_budget4510.npz`；
- OBSM--BFGS：`N_F=107268`。冻结 finalist 在早期保持同一算法和
  call-91 checkpoint；长程数值验证在 strong-Wolfe 无步长时，沿同一 paper-`p`
  方向采用逐次计费的 Armijo backtracking；若完整 image batch 仍未完成，则
  string 不移动、全部已花费调用保留，并把跨 outer 曲率状态重置到同一个 bonded
  `H0`。本次共记录 82 次曲率重启，最终在连续恢复上限处诚实停止；raw
  `runs/publication_long90k_20260802/publication_long90k_20260802_publication90k_robust_20260802r2_bfgs_lam0p3_inner1_r1_nocap_cfg9c7658b737c3_budget8000.npz`。

BFGS 的恢复是明确登记的 line-search/state-scope 实现选择，不是 IVM proximal，
也不把失败轮的 force cost 删除。它的长程 oscillation 在图中原样显示。正式图的
加密能垒数据分别为：

- `runs/diagnostics/dense_barrier_ala_original_dt0p014_publication90k.npz`；
- `runs/diagnostics/dense_barrier_ala_static_k30_dt0p013_publication90k.npz`；
- `runs/diagnostics/dense_barrier_ala_obsm_fire_lam0p85_inner3_dt0p1_publication90k.npz`；
- `runs/diagnostics/dense_barrier_ala_obsm_bfgs_lam0p3_robust_publication90k.npz`。

这些 dense diagnostics 单独计费并明确排除在 optimization 横轴之外；右图只需
显示到 30k，但超过 10 kcal/mol 的真实值仍保留在数据中，只由图上的上断轴裁出。

## 3. IVM 四层 clean-start 设计

共同 alpha 全网格为
`0.01, 0.1, 1, 10, 20, 30, 100, 1000, 10000, inf`。Stage 1/2 对
`r=0.2` 与 `r=0.1` 都跑完整网格；Stage 3/4 对每个 alpha 使用完全相同的
inexact policy。这里的顺序是有意的：一次只加入一类数值机制，不能把早期
pilot 的 alpha-specific trick 混入最终网格。

### Stage I：exact-like IVM，不 resample

- 固定 alpha；冻结 normal space 上的 proximal subproblem；projected L-BFGS。
- `r=0.2,0.1`；inner cap 80；history 7；proximal-only globalization。
- `reparam_mode=geometric` 但 `reparam_every=0`，所以完全不重参数化。
- 主实验 `budget=160`，`stop_iperp=-1`，因此 outer 层以 evaluation budget
  停止；单个 image 的 inner solve以相对 KKT 达标、80-step cap、保留的 outer
  evaluation boundary 或数值失败状态停止，详细向量保存在 NPZ。
- “exact-like”只表示相同高 cap 和较紧 KKT tolerance，不声称有限 nonlinear
  solve 达到解析 exact。
- 数据：`runs/stage1_exact_no_resample/`（20 个正式 r/alpha NPZ）。

### Stage II：exact-like IVM + resample

- 与 Stage I 完全相同的 `r=0.2,0.1`、inner cap 80 和 proximal-only policy。
- 唯一增加项：每个 outer step 后做 geometric equal-arclength resampling，
  `reparam_every=1`；新的路径必须重新调用 OpenMM，正常计费。
- 主实验 `budget=160`，`stop_iperp=-1`，实际以 evaluation budget 停止。
- 数据：`runs/stage2_exact_resample/`（20 个正式 NPZ）。

### Stage III：plain inexact IVM + resample

- 固定 alpha；每 image inner cap 5；`r=0.8`；proximal-only globalization。
- 同一 geometric equal-arclength resampling，`reparam_every=1`。
- 主实验 `budget=220`，`stop_iperp=-1`，实际以 evaluation budget 停止。
- 这是刻意朴素的 inexact baseline，不加入 production recovery 技巧。
- 数据：`runs/stage3_inexact_baseline_resample/`（10 个正式 NPZ）。

### Stage IV：uniform optimized inexact IVM + resample

所有 alpha 使用同一 two-phase state machine，且同一 scalar alpha 在两阶段保持
不变。最终正式短网格是 Phase-A34/Phase-B220；用于 5000-force alpha 图的长网格
沿用同一 policy，只增加 Phase-B budget 到 600/1000。

Phase A：

- base `r=0.96`，inner cap 15，history 7；相同的 stall tightening target 0.95
  对所有 alpha 生效；
- `iperp_monotone` state globalization，trust fraction 1.5，polish alpha factor
  0.5；
- `state_curve_projected` resampling 每 macrostep；direct trial cap 15；
  path-sum objective acceptance；
- 正式 transition 在 34-call Phase A 结束路径；该 terminal path 被保存为 continuation。

Phase B：

- 从 Phase A terminal Cartesian path 重启，清空 optimizer memory；初始化重新计费；
- `r=0.8`，inner cap 15，trust fraction 2；
- `state_curve` resampling 每 macrostep；per-image direct acceptance；
  CV trigger/strict trigger 均为 0.001，retry reduction 0.8；
- recovery cap 320，recovery inner cap 15，bicriteria merge，equidistance
  contraction 0.2；
- 运行级停机规则：只有完全 reevaluated、`reparam_committed` 的 state 达到
  `I_perp <= 1.5e-4` 才可停，否则用尽 Phase-B budget。alpha 20 A34/B220
  的记录为 `stop_reason=evaluation_budget`；其 terminal raw `I_perp` 虽略低于
  1.5e-4，但 terminal state 不满足该完整 committed 停机条件。

Stage-IV raw 数据：
`runs/stage4_inexact_optimized_resample/stage4_uniform_v3_a*_phaseA34_phaseB220.npz`。
主 wrapper 为 `src/run_stage4_uniform_optimized.py`，核心 runner 为
`src/run_ivm_v9_spring_projected_production_20260730.py`。

### Stage-IV v4 publication archive

publication trace 是统一的、reference-free、force-charged output policy：

1. raw `I_perp <= 0.007` 后激活；
2. 只接收完整 Phase-A projected state 或完整 Phase-B `reparam_committed` state；
3. 每个候选用 10 subdivisions 的 densified path screen，所有调用计费；
4. 相对 incumbent，`I_perp` 和 densified barrier value 均不得增加；
5. accepted incumbent 达到 `I_perp <= 1.9e-4` 后停止，否则穷尽 raw run。

它不读取 `Delta E*`，也不修改 IVM update；它是 accepted-iterate/output
globalization。20-subdivision reference error 只在选择完成后报告。archive 由
`src/build_pareto_archive.py` 通过 deterministic online-equivalent prefix replay
生成，`src/audit_stage4_v4_acceptance.py` fail-closed 审计。

## 4. 最终选择的整数 fixed-alpha IVM

最终选择：Stage IV v4 Pareto，`alpha=20`，两阶段始终为精确整数 20。

核心算法仍是论文的 IVM：每个 outer state 冻结当前 normalized tangent/normal
space，并在该 normal space 内求固定-alpha proximal subproblem；没有把 NEB
spring force、OBSM bare-energy solve 或动态 alpha 偷换进目标。projected L-BFGS
secant pairs 在当前冻结 normal space 内投影后复用；Phase B transition 明确清空
memory。resampling 是单独的 path-coordinate operation，提交前必须完整 reevaluate。

使用的数值实现技巧包括：固定的 per-image inner cap、run-local L-BFGS history、
path-sum/per-image 两阶段 acceptance、trust control、spacing-triggered
reparameterization、失败重试和 bicriteria recovery，以及 reference-free
complete-state Pareto output screen。所有 alpha 使用同一组技巧。

理论合规的准确表述：核心 update 符合 fixed-alpha frozen-normal proximal IVM；
但这是 inexact、有限 image、带可选 resampling 和 output archive 的 production
实现。因此可以称为“符合论文 IVM 方程的 inexact implementation”，不能声称它
自动继承论文中 exact unreparameterized iteration 的全部收敛定理。

最终 alpha-20 状态：

- accepted optimization call/cost：2507；
- Pareto screen：2301 force equivalents；
- total charged `N_F=4808`；
- raw `I_perp=1.8216907466e-4`；
- 20-subdivision densified barrier `2.4551593991 kcal/mol`；
- barrier absolute error `9.2311402e-4 kcal/mol`；
- selected path SHA-256：
  `009b6d129797a54b02f6c7424426ea24c76367a184f83e0066bfbcb9fae0bcff`；
- accepted calls：`660, 680, 829, 943, 2507`；accepted trace 两指标均零上升；
- 对 jointly eligible ODE--NEB：`I_perp=2.1e-4` 时 4808 vs 83522，
  `17.371x`；barrier error `0.017` 时 4808 vs 19222，`3.998x`。

权威代码和数据：

- core：`src/run_ivm_v9_spring_projected_production_20260730.py`；
- two-phase wrapper：`src/run_stage4_uniform_optimized.py`；
- raw combined：
  `runs/stage4_inexact_optimized_resample/stage4_uniform_v3_a20_phaseA34_phaseB220.npz`；
- archive：
  `runs/pareto_archive/pareto_clean_stage4_v4_a20_phaseA34_phaseB220.npz`；
- archive audit：同 stem 的 `.audit.json`；
- screen dense data：
  `runs/diagnostics/dense_barrier_clean_stage4_pareto_screen_a20_phaseA34_phaseB220.npz`；
- final audit：`reports/stage4_v4_fail_closed_acceptance.json`；
- 人类可读摘要：`reports/STAGE4_V4_ACCEPTANCE_SUMMARY.md`。

## 5. 四层实验的重要审计图（不全进论文）

- Stage I/II exact-like 两指标 2x4：
  `figures/clean_four_level_audit/alanine_clean_exact_alpha_two_metric_2x4.pdf`；
- Stage III/IV inexact 两指标 2x2：
  `figures/clean_four_level_audit/alanine_clean_inexact_alpha_two_metric_2x2.pdf`；
- 四层 alpha=20 总览：
  `figures/clean_four_level_audit/clear_four_groups_v4/alanine_clear_four_level_alpha20_overview.pdf`；
- 四组分开图：同目录的 `alanine_clear_stage{1,2,3}_two_metric.pdf` 和
  `alanine_clear_stage4_raw_and_accepted.pdf`；
- 生成器：`src/plot_clean_four_level_audit.py`、
  `src/plot_clean_four_groups_clear_v4.py`；
- machine audit：`figures/clean_four_level_audit/alanine_clean_four_level_audit.json`。
  `reports/clean_four_level_fail_closed_audit.json` 是早期 A80 raw-trace 的结构审计；
  它会诚实报告 raw barrier oscillation，不是当前 v4 acceptance verdict。当前
  fail-closed 结论只看 `reports/stage4_v4_fail_closed_acceptance.json`。

## 6. 正式论文图与复现登记

2026-08-04 重新直接审查 `paper/main.tex` 后，当前一共只有三个
Ala `\\includegraphics`，且全部指向
`paper/Figures/ala2_66d/official_20260802/`。图与 LaTeX label 的唯一对应是：

| 论文 label | 内容 | 稳定 PDF |
| --- | --- | --- |
| `fig:Ala2_accuracy` | IVM 与 ODE--NEB 的 1x4 路径演化 | `alanine_all_methods_1x4_path_evolution.pdf` |
| `fig:Ala2_loss_history` | 六种方法的两指标线性计费收敛与加速比 | `alanine_all_methods_two_metric_descent_linear_cost_speedup.pdf` |
| `fig:Ala2_alpha_mechanism` | Stage I 与 Stage IV 的固定 decade-alpha、共同 `N_F=5000` 轨迹 | `alanine_alpha_cost_to_precision_exact_vs_inexact.pdf` |

以下是这三个已采用 PDF 的详细复现登记：

1. `alanine_all_methods_1x4_path_evolution.pdf`
   - 内容：共同 IDPP、`N_F<=1k`、`N_F<=10k`、selected checkpoints；
   - 为保证几何路径可读性，只显示 IVM 和 ODE--NEB（定量图仍保留全部六种方法）；
   - plotter：`src/plot_stage4_v4_pareto_all_methods.py`；
   - source figure SHA-256：
     `cb8c929a42ae1bf4a389dab7fc9a62f262f19432ca7267ca7bcbc062b185c055`。
2. `alanine_all_methods_two_metric_descent_linear_cost_speedup.pdf`
   - 内容：linear `N_F` 横轴，`I_perp` 与 densified `Delta E`，标注
     左图到 90k 且纵轴下限为 `5e-5`；所有非 IVM raw histories 都真实超过
     90k。右图到 30k，纵轴显示 `2--10 kcal/mol`，在 10 处用锯齿上断轴
     表明仍保留更大的真实值；
     标注 `17.4x` 和 `4x` 均置于对应黑色线段下方；
   - plotter：`src/plot_all_methods_linear_speedup_v5.py`；
   - audit：
     `figures/clean_four_level_audit/all_methods_stage4_v4_pareto_linear_speedup_v5/LINEAR_SPEEDUP_AUDIT.json`；
   - SHA-256：
     `df652cdecc553d12d951ff8c15ad305fb7087454567b9825a2a15bb753095e1a`。
3. `alanine_alpha_cost_to_precision_exact_vs_inexact.pdf`
   - 内容：Stage I `r=0.1` 无 resampling vs Stage IV
     `r=0.96 -> 0.8` + resampling；只保留 decade alpha；所有曲线截到共同
     charged `N_F=5000`；画的是每个完成态的 uncensored raw `I_perp`
     轨迹，不使用 best-so-far 前沿或“last improvement 后继续计费”的虚线；
   - plotter：`src/plot_alpha_common5000_decades_retitle_v10.py`；它复用
     `plot_alpha_common5000_decades_v9.py`、`plot_alpha_common5000_v8.py` 与
     `plot_alpha_full_budget_v6.py` 的读取/画图函数，这些文件是依赖而非正式版本；
   - audit：
     `figures/clean_four_level_audit/alpha_common5000_decades_v10_titles/TITLE_PROTOCOL_AUDIT.json`；
   - SHA-256：
     `2b5d617935a9706076354e90fe6d9c88ac2f92a254b2ede3990f125fbb58385a`。

高层审查副本位于
`paper/ActiveExperiments/alanine_66d/figures/`，三个文件名和 SHA-256
均与上述 official PDF 一致。这个高层目录是为了审查、导航和冻结，
不是新的科学结果谱系；`main.tex` 仍保持引用 `official_20260802/`
的稳定路径。

复现命令（不会重跑 dynamics，只从冻结数据重画）：

```bash
cd /Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python src/plot_stage4_v4_pareto_all_methods.py
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python src/plot_all_methods_linear_speedup_v5.py
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python src/plot_alpha_common5000_decades_retitle_v10.py
```

然后只把相应最终 PDF 复制到 `official_20260802/` 的稳定文件名；不要让
`main.tex` 引用带 `v5/v10` 的中间目录。

## 7. 归档规则

- `paper/ActiveExperiments/alanine_66d/`：唯一 Ala 启用版本入口；
- `paper/ExperimentArchive/`：论文侧的过时实验 bundle 总归档区，
  不得被解读为当前结论；
- `paper/Figures/ala2_66d/archive_legacy_pre_clean_20260802/`：旧论文 PDF、旧
  manuscript candidates 和 pre-clean figure bundles；
- `paper/Figures/ala2_66d/clean_*`：最终 official PDF 的构建输出和中间 candidate；
  它们只能用于追溯，不能仅因时间戳更新就替代 official PDF；
- `figures/archive_obsolete_pre_final_20260802/`：v5--v9 中间图和被最终 v10/v5
  替代的 plot outputs；
- `runs/stage4_policy_pilots/`、`logs/stage4_*pilots/`、
  `reports/stage4_phaseb_policy_pilots*.json`：失败/未采用的 policy pilot，保留作
  negative evidence，不得混入最终 alpha grid；
- 旧 plotter 若仍被最终 plotter import，则不能移动或删除；它们是代码依赖，
  不是正式结论。

任何新的 dynamics、acceptance rule、alpha-specific trick 或计费变更都必须新建
版本标签、重跑完整共同 alpha grid、更新本文件与 machine audit，不能覆盖当前
冻结结果。
