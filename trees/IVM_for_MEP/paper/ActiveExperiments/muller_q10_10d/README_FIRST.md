# Q-transformed 10D Muller-22 实验总说明：审查前必须先读

状态：启用实验谱系，按 2026-08-04 的真实代码、冻结配置和结果整理。

> **2026-08-20 组合图登记：** 当前 `N_int=10`（12 total images）与
> `N_int=20`（22 total images）的 2x3 candidate 图，其实验代码、冻结 NPZ、
> selection reports、contour checkpoints、source CSV、manifest 和唯一重绘命令统一登记在
> `FIGURE_N10_N20_2X3_PROVENANCE.md`。审查该组合图时，以该专用 provenance 为入口。

本文件是 Q-transformed、无正弦余弦扰动、10 维 Muller-Brown、22 images
算例的最高优先级说明。审查代码、引用数字、选择图片、重跑实验或继续调参前，
必须先读本文件。不要根据 working folder 中某个 `screen`、`candidate`、
`provisional`、旧 `FINAL_REPORT.md` 或目录时间戳反推当前结论。

本文件只描述实际执行过并留下代码或冻结结果的实验。用户口述与真实执行有出入时，
以真实代码、NPZ/JSON 配置和 force ledger 为准。最重要的纠正是：ASE 的合法输入是
`NEBOptimizer(neb, method="ode")`，不是 `method="pde"`。代码中没有 `pde`
optimizer。

当前实验谱系由以下入口共同构成：

1. 论文侧启用快照：`paper/ActiveExperiments/muller_q10_10d/`。
2. 完整 working 谱系：`/Users/wuzhiyou/Downloads/IVM_for_MEP/muller_q10_ivm/`。
3. 论文源唯一真值：`/Users/wuzhiyou/IVM_for_MEP/paper/main.tex`。
4. 当前论文实际引用的 Muller PDF：`/Users/wuzhiyou/IVM_for_MEP/paper/Figures/Muller10/`。

论文侧 `code/` 是启用代码的冻结审阅快照，`data/` 保存关键输入、报告和选中结果，
`figures/` 区分当前论文实际引用与最新已审计 candidate，`archive_legacy/` 保存退出
启用谱系的旧说明、provisional 图和本次不再研究的 N10 图。重跑时优先从完整 working
目录执行，因为原 plotter 和 runner 使用相对于 working case 根目录的数据路径。

## 0. 当前启用范围和不可混淆事项

- 体系是 10D Q-transformed Muller-Brown，不带 sine/cosine 或其他 roughness 扰动。
- 路径是 22 个 images，其中 20 个为可动内部 images。本快照不重新研究 12-image case。
- 所有方法使用 Henkelman-Jonsson/ASE improved tangent。
- 所有正式横轴 `Number of Forces` 都是 20 个内部 images 的 uncached force calls 直接求和。
- 固定端点不计优化 force cost；被拒绝的 trial force call 必须计费；cache hit 不重复计费。
- 旧 `reports/BASELINE_SELECTION.md` 中“横坐标为 20 个 image 的最大 calls”是早期表述，
  已被后续 force audit 和本文的直接求和定义取代。
- 传统 22-image `I_perp` 与后期 5001 点 densified `I_perp` 是不同指标，不能交叉比较数值。
- `main.tex` 目前仍同时引用 N10 与 N20 旧 official 图。本目录的科学范围只有 N20。

## 1. 共同势能、Q、初始路径和指标

对列向量使用

```text
z = Q @ x
U(x) = U_M(z_1, z_2) + 2 * sum_{j=3}^{10} z_j^2
grad_x U = Q.T @ grad_z U
force_x = -grad_x U
```

`Q` 由 `numpy.random.default_rng(0)` 生成一个 `10 x 10` iid 标准高斯矩阵，
进行 QR 分解，并作确定性的列符号归一化使 `R` 的对角非负。冻结输入为
`data/inputs/Q_data.npz`，SHA-256 为
`372962a47504e6df28a20d8b022482a6835680765434b4385a6dba0e826974bc`。

初始路径先在自然 `z` 坐标中连接两个 Muller minima，再逐点用 `x = Q.T @ z`
变换到物理 `x` 坐标。势能、解析力、inactive harmonic force、端点变换和中心差分
检查全部通过，审计记录为 `data/reports/q10_potential_audit.json`。

传统正式比较保存两类指标：

- 离散 `I_perp`：22-image 路径上的法向残差积分。
- 加密 barrier：按正式 Muller plotter 对路径做 coordinate-wise cubic densification 后计算。

后期 NN-IVM 比较统一使用 5001 点证书：coordinate-wise not-a-knot cubic spline、
5001 点、同一个 improved tangent，再计算 densified `I_perp` 和
`|Delta E - Delta E*|`。5001 点证书只做离线诊断，不进入优化、停机或配置切换。

## 2. Baseline 总体搜索和选择原则

主要 baseline 共五类：Original String、OBSM-FIRE、OBSM-LBFGS、Static NEB、
ODE-NEB。全局初筛实际运行 1486 个配置，初筛 budget 为每个内部 image 250 calls。
随后重跑 38 个 shortlisted 配置；targeted refinement 又运行 272 个配置，其中
OBSM-FIRE 40 个、OBSM-LBFGS 90 个、Static NEB 52 个、旧 ODE refinement 90 个。
最终候选按 trajectory score、terminal `I_perp` 和 dense barrier 分别保留并长预算重跑。
canonical ODE 最后另做只调 `k` 的 ASE-default sweep，覆盖旧的 ODE `rtol/C2` 联合网格。

参数只在同一种方法内部排序，没有利用 IVM 的结果替 baseline 挑参数。Original String
和 OBSM 的重参数化都必须调用 `code/baselines/run_q10_baselines.py` 中
`appendix_cubic_spline`/`cubic_arclength_reparameterize`，这是按 Simplified String
Method Appendix A 实现的自定义 cubic spline，不是线性插值，也不是随意换成 SciPy
默认 spline。

### 2.1 Original String

算法：所有内部 images 沿 full physical force 做显式 string/Euler 更新，按设定 interval
用 Appendix-A cubic spline 重新分布到等弦长 labels。它没有 OBSM 的冻结 normal
子问题，也没有 IVM proximal 项。

真实参数范围：

- `dt = 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2`。
- `reparam_interval = 1, 2, 5`。
- 长预算 refinement 重点覆盖小 `dt`，避免大步长早期下降掩盖后期误差。

选中配置：`dt=3e-5`，`reparam_interval=2`，diagnostic interval 25。

选中结果：direct total forces `99520`，离散 `I_perp=5.918709e2`，dense barrier
error `6.046293e-6`。它的 barrier 很准，但法向残差很差，不能称为整体收敛。

代码：`code/baselines/run_q10_baselines.py`。

冻结结果：`data/frozen_runs/original_string_selected.npz`。

### 2.2 OBSM-FIRE

算法：在当前 improved tangent 定义的冻结 normal hyperplane 上，对每个 image 用 FIRE
近似极小化裸势能 `U`，不含 IVM 的 `||delta||^2/(2 alpha)` proximal 项。inner 解后用
`x_new=(1-lambda)x_old+lambda*x_min` 混合，再按 interval 使用同一个 Appendix-A
cubic spline 重参数化。FIRE 维护 velocity，使用显式 force 更新和冻结 normal 投影。

真实参数范围：

- `lambda = 0.1, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99`。
- `inner = 2, 4, 8, 16, 32`。
- `fire_dt = 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 5e-3, 1e-2`。
- `fire_beta = 0.1, 1.0`。
- `reparam_interval = 1, 5`。

选中配置：`lambda=0.8`，4 个 inner FIRE steps，`fire_dt=0.01`，
`fire_beta=1`，`reparam_interval=5`。

选中结果：direct total forces `7252`，离散 `I_perp=2.866764e-23`，dense barrier
error `1.885190e-3`，stationary stop。它是传统 baseline 中最强的 residual competitor。

代码：`code/baselines/run_q10_baselines.py`，通过 `solver="fire"`。

冻结结果：`data/frozen_runs/obsm_fire_selected.npz`。

### 2.3 OBSM-LBFGS

算法外层与 OBSM-FIRE 相同，inner solver 改为 reduced frozen-normal coordinates 上的
SciPy `L-BFGS-B`。这里没有 active bounds，所以数值行为是无约束 L-BFGS；名称保留为
OBSM-LBFGS。它仍然极小化裸势能，不是 IVM proximal objective。每轮同样进行
`lambda` 混合和 Appendix-A cubic reparameterization。

真实参数范围：

- `lambda = 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99`。
- `inner = 1, 2, 4, 8, 16`。
- `reparam_interval = 1, 2, 5`。

选中配置：`lambda=0.2`，2 个 inner L-BFGS steps，`reparam_interval=1`。
早期看起来很强的 `lambda=0.25` 长程发生 rebound/numerical failure，因此没有选。

选中结果：direct total forces `15530`，离散 `I_perp=5.229002e-22`，dense barrier
error `1.885190e-3`，stationary stop。

代码：`code/baselines/run_q10_baselines.py`，通过 `solver="lbfgs"`。

冻结结果：`data/frozen_runs/obsm_lbfgs_selected.npz`。

### 2.4 Static NEB

算法严格使用 ASE `NEB` 数据结构和 ASE optimizer：

```python
neb = NEB(images, k=k, method="improvedtangent", ...)
optimizer = NEBOptimizer(neb, method="static", alpha=alpha)
```

我们没有自己实现 static NEB optimizer。`k` 是 spring constant，`alpha` 是 ASE
static optimizer 的显式步长。

真实参数范围：

- `k = 0.1, 0.3, 1, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500`。
- `alpha = 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2`。
- `fmax=1e-14` 只用于避免宽松的 optimizer convergence 提前终止。

选中配置：`k=200`，`alpha=3e-4`，diagnostic interval 10。弱 spring 的 residual
早期可能更快，但路径 spacing 和 barrier 更差；最终按 joint path-quality compromise 选择。

选中结果：direct total forces `99820`，离散 `I_perp=4.229283e-8`，dense barrier
error `1.312659e-3`。

代码：`code/baselines/run_q10_baselines.py`，targeted grid 为
`code/baselines/run_q10_refinement.py`。

冻结结果：`data/frozen_runs/static_neb_selected.npz`。

### 2.5 ODE-NEB

算法严格使用 ASE `NEB` 和 ASE ODE optimizer：

```python
neb = NEB(images, k=k, method="improvedtangent", ...)
optimizer = NEBOptimizer(neb, method="ode")
```

真实输入是 `method="ode"`，不是 `pde`。我们没有另写 ODE-NEB，也没有覆盖 ASE
ODE constructor 的 `alpha`、`verbose`、`rtol`、`C1` 或 `C2` defaults。canonical
实验只调 `k`。

canonical `k` grid：`0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000`。
选择标准是在总 force budget 5000 内，先满足可接受 barrier，再比较 joint endpoint
tradeoff 和 early-window behavior。`k=20` 的确早期很快，但在约 1000 total forces 后
rebound，因此没有选。最终选择 `k=1000`。

选中结果：长程 direct total forces `99960`，离散 `I_perp=1.807288e-11`，dense
barrier error `1.888222e-3`。5000-total-force 选择 checkpoint 为约 `4940` forces，
`I_perp=1.433479e-5`，barrier error `3.001101e-4`。

代码：`code/baselines/run_q10_ode_default_k_sweep.py` 和
`code/baselines/run_q10_baselines.py`。

冻结结果：`data/frozen_runs/ode_neb_k1000_selected.npz`。

### 2.6 Baseline 最终表

| 方法 | 最终配置 | Direct total forces | 离散 `I_perp` | Dense barrier error |
|---|---|---:|---:|---:|
| Original String | `dt=3e-5`, cubic every 2 | 99520 | `5.918709e2` | `6.046293e-6` |
| OBSM-FIRE | `lambda=0.8`, inner 4, `dt=0.01`, `beta=1`, cubic every 5 | 7252 | `2.866764e-23` | `1.885190e-3` |
| OBSM-LBFGS | `lambda=0.2`, inner 2, cubic every 1 | 15530 | `5.229002e-22` | `1.885190e-3` |
| Static NEB | ASE static, `k=200`, `alpha=3e-4` | 99820 | `4.229283e-8` | `1.312659e-3` |
| ODE-NEB | ASE ode defaults, only `k=1000` tuned | 99960 | `1.807288e-11` | `1.888222e-3` |

完整筛选记录：`data/reports/baseline_screen_summary.json`、
`data/reports/baseline_final_selection.json` 和 `data/reports/q10_ode_default_k_sweep.json`。

## 3. 四组 fixed-alpha IVM 的整体实验逻辑

四组实验不是漫无目的地堆数值技巧。顺序是：先检查理论 proximal map，再只加入
重参数化，再把 inner solve 改为朴素 inexact，最后加入统一且 alpha-independent 的
历史、预条件和 trust policy。四组都固定 alpha；每个 alpha 从干净状态启动，不共享
位置 cache 或 optimizer memory。

Groups 1-3 的共同 alpha grid 为
`1e-4, 3e-4, 1e-3, 1.3e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 100, 1000, 10000`。

### 3.1 Group 1：exact IVM，不 resample

- 冻结当前 improved tangent 和完整 9D normal space。
- 对每个 image 用 multistart L-BFGS-B 近乎精确极小化原始 proximal objective。
- `r=kkt_rtol=0.2, 0.1, 0.05`，并固定 `kkt_atol=1e-3`。
- `r` 是相对 KKT certificate，不是 optimizer gradient tolerance。
- 无 resample，无 spring；配置中保留的 spring 字段不被 `resample_mode="none"` 调用。
- inner gradient tolerance `1e-10`，outer limit 80，budget 5000 calls/image。
- outer 停机为 movement `<=1e-14`、budget、outer limit 或明确数值失败。

三个 `r` 的结果数值上几乎重合，因为 inner solve 比所有证书要求更精确。正式展示用
`r=0.1` 作为中间、代表性选择，不声称它比 `r=0.05` 更“精确”。

代码：`code/ivm/run_q10_ivm_groups.py`、`code/ivm/q10_ivm_core.py`。

数据：`data/reports/q10_exact_no_resample_summary.json`。

### 3.2 Group 2：exact IVM + cubic resample

- 与 Group 1 完全相同的 alpha grid、multistart solver 和 `r=0.2,0.1,0.05`。
- 唯一新增机制是每个 outer iteration 后做 Simplified-String cubic resample。
- 每次 resampling 做 3 个 cubic passes。
- improved-tangent spring pilot 实际更差，正式 Group 2 不使用 spring reparameterization。
- 停机条件与 Group 1 相同。

目的：单独判断 equal-arclength cubic resample 是否能避免 images 分布不均，而不把
inexact solver 或 history 混进来。正式展示仍取 `r=0.1`。

代码：`code/ivm/run_q10_ivm_groups.py`、`code/ivm/q10_ivm_core.py`。

数据：`data/reports/q10_exact_resample_summary.json`。

### 3.3 Group 3：plain inexact IVM + cubic resample

- 固定 alpha，完整 9D normal subproblem。
- 每个 image 每个 outer 固定接受 2 个 full-normal L-BFGS steps。
- 每个 subproblem 最多 12 个新 force calls。
- 不使用 `r`，因为停机规则已经改为固定 inner-step budget。
- history size 0，不跨 outer warm start，不使用 harmonic preconditioner。
- Armijo `1e-4`，backtracking `0.5`，`max_step=3`。
- 每个 outer 做 3 个 cubic passes，resample 后清空 history。
- budget 2000 calls/image，movement tolerance `1e-14`。

目的：得到一个有 resample 但没有 production acceleration trick 的诚实 inexact baseline。

代码：`code/ivm/run_q10_ivm_groups.py`、`code/ivm/q10_ivm_core.py`。

数据：`data/reports/q10_inexact_baseline_resample_summary.json`。

### 3.4 Group 4：统一调优的 inexact IVM + cubic resample

- 每个 image 每个 outer 固定 1 个 accepted full-normal L-BFGS step。
- 不使用 KKT ratio `r`；固定一步就是 inexact stopping rule。
- 每个 subproblem 最多 12 个新 force calls。
- physical-potential L-BFGS history size 15。
- 保存 `s=x_{k+1}-x_k` 和 `y_U=grad U(x_{k+1})-grad U(x_k)`。
- 新 normal basis `B` 中使用 `B.T@s` 与 `B.T@y_U+(B.T@s)/alpha`，只保留正曲率 pairs。
- 八个 inactive transformed coordinates 使用精确 harmonic preconditioner。
- 前 5 个 outer 的 `max_step=0.065`，之后固定 `max_step=0.5`。
- Armijo `1e-2`，backtracking `0.5`。
- 每 5 个 outer 固定做 1 次 cubic resample，resample 后保留 physical `U` secants。
- budget 700 calls/image，movement tolerance `1e-14`。
- 所有 alpha 使用完全相同 policy；没有 alpha-dependent trick。

实际运行 23 个固定 alpha，从 `1e-4` 到 `1e4`，并在 `1e-3` 附近及 inexact
稳定边界加密。完整取值以 `data/reports/q10_inexact_tuned_resample_summary.json`
为准。最终选择标准十进制值 `alpha=1e-3`；旧 `1.25e-3` 只保留为 tuning result。

代码：`code/ivm/run_q10_tuned_final_grid.py`、`code/ivm/q10_ivm_core.py`。

数据：`data/reports/q10_inexact_tuned_resample_summary.json`。

### 3.5 四组重要图

- Alpha 总览：`figures/active_candidates/q10_four_group_alpha_overview.pdf`。
- 四个代表配置直接比较：`figures/active_candidates/q10_four_group_selected_comparison.pdf`。
- 绘图代码：`code/plotting/plot_q10_alpha_studies.py`、
  `code/plotting/plot_q10_four_group_comparison.py`。

## 4. 最终选中的传统 IVM

最终传统 IVM 是 Group 4、固定 `alpha=1e-3`。每个 outer 冻结 improved tangent 和
normal basis，在完整 9D normal space 内近似极小化

```text
U(phi_k + P_k delta) + ||P_k delta||^2 / (2 alpha)
```

`delta` 是无约束变量。没有 Lagrangian、NEB spring force、OBSM bare-energy solve、
iterate mixing、Anderson acceleration、alpha schedule 或输出 clipping。

数值 trick 只有对同一个 proximal 子问题合法的 inexact realization：一步 L-BFGS、
physical-potential secant history 的 basis reprojection、已知 harmonic block 的
preconditioner、固定 trust schedule、proximal Armijo，以及每 5 outer 一次 cubic
resample。每个 accepted update 都通过当前 proximal objective nonincrease 审计。

停机规则：movement tolerance `1e-14`；若未达到则受每 image 700 calls budget 和 outer
limit 限制。选中运行在每 image 最大 132 calls 时 stationary，direct total forces
为 `2377`。

选中结果：离散 `I_perp=2.332795e-23`，dense barrier error
`1.885189576e-3`，segment ratio `1.000000000000130`，最大 accepted proximal
increase `9.622876e-31`。它比 OBSM-FIRE 更早到达多个 residual threshold，且终端
离散 residual 更小。

理论一致性结论：符合 fixed-alpha frozen-normal IVM 的 inexact solve。需要诚实说明，
固定一步 realization 不继承 exact implicit proximal map 对任意大 alpha 的 unconditional
stability；这解释了 `alpha>=1.4e-3` 附近的 inexact deterioration，不反驳 exact 理论。

代码：`code/ivm/run_q10_tuned_final_grid.py`、`code/ivm/q10_ivm_core.py`。

冻结结果：`data/frozen_runs/tuned_ivm_alpha1e3_selected.npz`。

后期 5001 点 common certificate 下，这条 22-image 离散 IVM 的 densified
`I_perp` 为约 `509.32`。这不是与 `2.33e-23` 矛盾，而是连续 cubic interpolation
在 images 之间仍有 residual；这个观察正是继续做 NN-IVM/连续路径实验的原因。

## 5. NN-IVM 与连续表示实验

所有启用的后期 NN/连续表示实验固定 `alpha=1e-3`，每轮使用 20 个移动 samples，
使用 improved tangent，并对 projected displacement `delta` 做无约束优化。没有
Lagrangian 或 barrier penalty。`torch-nightly` 用于 PyTorch 网络代码；最终 cubic
B-spline runner 的子问题由 SciPy L-BFGS-B 求解。

### 5.1 实际探索的结构

| 结构 | 实际主要范围 | 可训练参数/系数 | 子问题 optimizer | 结论 |
|---|---|---:|---|---|
| Full MLP | width 8/16/24，depth 1/2；正式审计 width 16, depth 2 | 474 | L-BFGS-B，正式 inner 3 | 不稳定，淘汰 |
| Tanh random feature | width 8/16/32/64，冻结 hidden，只训 readout | 约 `10*(width+1)` | L-BFGS-B | 未胜出 |
| Random Fourier feature | width 8/12/16/24，bandwidth 1/3/6 | 约 `10*(width+1)` | L-BFGS-B | 未胜出 |
| Gaussian RBF random feature | width 8-24，sigma 0.04-0.16；正式 width 18, sigma 0.08 | 190 | L-BFGS-B，正式 inner 3 | 比 full MLP 稳，但不够准 |
| Endpoint-exact pointwise RBF | width 100-240，overlap scale 1.5-3.0；正式 width 160 | 1600 linear coefficients before projection | sampled scalar proximal/secant + compression | 曾小幅胜过 discrete IVM，后被 spline 取代 |
| Fourier/local hybrid basis | 多个 Fourier/local widths、sampling policies | 随 width 变化 | L-BFGS-B | 没有超过最终 spline |
| Cubic B-spline continuous path | production width 12 -> 14 -> 8 | 120 -> 140 -> 80 | L-BFGS-B，inner 3 | 最终胜出 |

Full MLP 的 474 个参数来自 `1->16`、`16->16`、`16->10` 三个带 bias 的 linear
layers。RBF RF 的 190 个参数是固定 18 个 features 后的 `18->10` readout 加 bias。
Cubic B-spline 的系数维数为 `width*10`。所有结构用 endpoint envelope 或等价构造
保持端点固定，并将输出投影到当前 frozen normal subspace。

探索过的统一配置集合还包括 `inner_iterations=1,2,3,5`、RBF sigma
`0.04,0.06,0.08,0.10,0.12,0.16`、RF bandwidth `1,3,6`。相关 working runs
总计至少 77 个 MLP/RF audit 配置，另有 pointwise、constrained-theta、normal-stack、
two-chart、residual 和 spline tuning 谱系。它们没有全部复制到启用快照；原始数据保留
在 working `runs/`，本文件只冻结最终可审阅代码和代表 artifact。

### 5.2 Full MLP 与 random-feature 结果

Full MLP 正式审计配置：width 16、depth 2、474 trainable parameters、fixed network
seed、arc-length reparameterization、20 samples、L-BFGS-B inner 3、force budget 3500。
其最小 densified `I_perp=1.59115e4`，终端约 `7.57020e6`，明显失败。

RBF random-feature 正式审计配置：width 18、sigma 0.08、固定 RBF features、只训练
190-parameter readout、无额外 arc map、L-BFGS-B inner 3、force budget 3500。
其最小 densified `I_perp=1.38975e3`，终端约 `2.98753e3`，也未通过 barrier gate。

对应代码：`code/nn_ivm/q10_nn_ivm.py`、`code/nn_ivm/q10_nn_ivm_mlp_lbfgs.py`。

对应冻结结果：`data/frozen_runs/full_mlp_selected_audit.npz`、
`data/frozen_runs/rbf_random_feature_selected_audit.npz`。

### 5.3 早期 pointwise RBF NN-IVM

一个曾经的 production 使用 width-160 endpoint-exact Gaussian RBF、20 维
unscrambled Halton stratified offsets、jitter 0.495、每 sampled scalar frozen-normal
proximal subproblem 最多 3 calls、trust radius 0.03、relative criterion `r=0.8`，
再做 513-point zero-force equal-arc teacher/compression。它在 2975 forces 得到
densified `I_perp=434.57` 和 barrier error `9.81e-5`，比当时 discrete IVM 的
densified residual 约好 15%，但没有数量级优势，因此退出最终 production。

代码：`code/nn_ivm/q10_nn_ivm_pointwise.py`。

完整旧说明保留在 `archive_legacy/Q10_NN_IVM_EXPERIMENTS.md`。

### 5.4 最终 cubic-spline continuous IVM

项目文件仍使用 “NN-IVM” 标签，但最终 winning representer 是 cubic B-spline，
不是多层全连接神经网络。它仍是连续参数化的 fixed-alpha IVM realization。

锁定 schedule：

| Direct total-force interval | B-spline width | Improved-tangent stencil | Inner L-BFGS iterations |
|---:|---:|---:|---:|
| 0-3200 | 12 | `0.002` | 3 |
| 3200-3700 | 14 | `0.001` | 3 |
| 3700-4000 | 8 | `0.0001` | 3 |

共同设置：fixed `alpha=1e-3`、20 moving samples、deterministic golden-shift
collocation、unconstrained projected delta、improved tangent、每轮 arc-length
reparameterized continuous path。优化目标始终只有 sampled proximal objective；
没有 dense feedback、Lagrangian、mix、trust clipping、output clipping 或 barrier penalty。

生产结果：direct total forces `4000`，5001 点 densified `I_perp=4.1401243274`，
minimum `4.1010652562`，barrier error `1.5818082204e-6`，maximum inactive
coordinate `6.37e-12`。三个 nominal seeds 得到完全相同结果，因为 golden-shift
policy 是确定性的；这是 deterministic reproducibility，不是随机鲁棒性证据。

与各 baseline 最后一个 `<=5000` total-force common-certificate checkpoint 比，
最好的非 spline densified `I_perp` 约为 `509.32`，所以 spline 约低 123 倍。
它的 final barrier error 也低于所有 baseline 的 final value；但若只挑 barrier 曲线
上的孤立 transient minimum，部分 baseline 曾低于 `1.58e-6`。诚实结论是同一
checkpoint 的联合优势，不是 barrier 单指标逐点全程支配。

代码：`code/nn_ivm/q10_nn_ivm_fourier.py`、
`code/nn_ivm/run_q10_spline_joint_production.py`。

冻结数据：`data/frozen_runs/spline_stage1.npz`、`spline_stage2.npz`、
`spline_stage3.npz` 及对应 audit JSON。

完整审计：`data/reports/Q10_CUBIC_SPLINE_JOINT_AUDIT.md`、
`data/reports/q10_spline_joint_production_locked.json`。

### 5.5 NN/连续表示图与 plotter

- Spline 与全部主要方法的 common-certificate 图：
  `figures/active_candidates/muller22_q10_nn_ivm_dense_comparison.pdf`。
- Full MLP、RBF RF、Spline 与 discrete IVM：
  `figures/active_candidates/muller22_q10_mlp_random_feature_study.pdf`。
- 连续 spline path evolution 1x4：
  `figures/active_candidates/muller22_q10_nn_ivm_path_evolution_1x4.pdf`。
- 统一 dense 后处理：`code/plotting/postprocess_q10_dense_comparison.py`。
- 主 common-certificate plotter：`code/plotting/plot_q10_nn_ivm_dense_comparison.py`。
- 网络结构 plotter：`code/plotting/plot_q10_mlp_rf_study.py`。
- 连续路径 plotter：`code/plotting/plot_q10_nn_ivm_path_evolution_1x4.py`。

## 6. 正式论文图片登记

必须区分“`main.tex` 当前实际引用”和“最新审计后建议启用”。截至 2026-08-04，
`main.tex` 在 Muller subsection 实际引用：

- `Figures/Muller10/N10_BestMethods_2x2_lossperp_lossarc_deltaE_contour_gradEvals_compressed.pdf`。
- `Figures/Muller10/N20_BestMethods_2x2_lossperp_lossarc_deltaE_contour_gradEvals_compressed.pdf`。

本目录只负责 N20。当前被 `main.tex` 引用的 N20 PDF 冻结为
`figures/currently_cited/N20_BestMethods_2x2_current_main.pdf`，SHA-256 为
`f1ee20073892804a8fe2b5d01c359626aa793a48097d216d91a9e9f976fec9ab`。

working folder 最新 formal rebuild 冻结为
`figures/active_candidates/muller22_q10_article_method_comparison_latest.pdf`，SHA-256 为
`11d9c198655ce2acf27012eb3280641cd02321c3af14864e0de051de6bf3340a`。
这两个 hash 不同，所以最新 candidate 尚不能冒充 `main.tex` 当前图。本文档没有修改
`main.tex`，也没有覆盖 `paper/Figures/Muller10/`。

建议的 N20 图分级：

| 级别 | 图片 | 状态 | 绘图代码 |
|---|---|---|---|
| 当前正文 | `figures/currently_cited/N20_BestMethods_2x2_current_main.pdf` | `main.tex` 正在引用 | 历史 formal notebook |
| 最新传统方法 candidate | `figures/active_candidates/muller22_q10_article_method_comparison_latest.pdf` | 已审计，尚未进 `main.tex` | `code/plotting/plot_q10_article_muller22.py` |
| 传统方法 1x3 linear-3000 candidate | `figures/active_candidates/muller22_q10_article_method_comparison_1x3_linear3000.pdf` | 已登记，尚未进 `main.tex` | `code/plotting/plot_q10_article_muller22_1x3_linear3000.py` |
| 四组方法学图 | `figures/active_candidates/q10_four_group_selected_comparison.pdf` | 建议 supplement/method audit | `plot_q10_four_group_comparison.py` |
| Alpha 总览 | `figures/active_candidates/q10_four_group_alpha_overview.pdf` | 建议 supplement | `plot_q10_four_group_comparison.py` |
| 路径演化 | `figures/active_candidates/q10_muller_path_evolution_1x4.pdf` | active candidate | `plot_q10_path_evolution_1x4.py` |
| Spline NN-IVM 主比较 | `figures/active_candidates/muller22_q10_nn_ivm_dense_comparison.pdf` | active candidate，尚未进正文 | `plot_q10_nn_ivm_dense_comparison.py` |
| 网络结构对比 | `figures/active_candidates/muller22_q10_mlp_random_feature_study.pdf` | architecture audit | `plot_q10_mlp_rf_study.py` |
| Spline path evolution | `figures/active_candidates/muller22_q10_nn_ivm_path_evolution_1x4.pdf` | continuous-path audit | `plot_q10_nn_ivm_path_evolution_1x4.py` |

### 6.1 `muller22_q10_article_method_comparison_1x3_linear3000.pdf` 完整溯源

本小节登记 2026-08-20 生成的传统方法 `1x3` active candidate。它不是新一轮实验，
而是对本文件第 2 节和第 4 节已经选定并冻结的六个 run 重新排版。当前 PDF 为
`figures/active_candidates/muller22_q10_article_method_comparison_1x3_linear3000.pdf`，
SHA-256 是
`841c673774e6243b850bb2c7d56a9763dd75d4d0b502452f299a1d4c8b25e9c4`。
该图尚未写入 `main.tex`，也没有覆盖 `paper/Figures/Muller10/`。

图的固定展示规则如下。

| Panel | 内容 | 数据字段与定义 |
|---|---|---|
| (a) | 离散 `I_perp` | 从每个 checkpoint 的 `normal_force_history` 计算正式 Muller plotter 的 `loss_perp`；tangent 为 improved tangent |
| (b) | Densified energy barrier `Delta E` | 直接读取每个 NPZ 的 `delta_e`；该值是在 checkpoint 路径上做 500 点 coordinate-wise cubic interpolation 后计算的加密能垒误差 |
| (c) | 最终路径 | 读取每个 NPZ 的最后一个 `path_history`，用固定 `Q` 映回 Muller `z_1,z_2` 平面并叠加正式等高线 |

横坐标读取 NPZ 中的 `total_force_evaluations`，含整条 22-image 路径的全部 force
calls，不是单 image 计数，也不再额外乘 20。横坐标为线性尺度，固定范围
`0 <= Number of Forces <= 3000`，主刻度为 `0,1000,2000,3000`。超过 3000 的
原始 checkpoint 只在视窗中被裁掉，NPZ 没有截断或改写。原 2x2 图的 `I_arc`
panel 被删除，剩余 panel 重新编号为 `(a),(b),(c)`。方法顺序和正式配色为 IVM、
ODE-NEB、Static NEB、Original String、OBSM-FIRE、OBSM-LBFGS，IVM 保持最高
绘制层级。画布尺寸为 `180 mm x 70 mm`，三个 panel 等宽，并通过
`axis.set_box_aspect(1.0)` 固定为 `1:1` 的近正方形 axes；该版式直接覆盖同名旧图，
不保留长方形 panel 版本。Panel letters 保持在各 axes 左上方外部；legend 与 panel
之间保留紧凑但不贴合的间距。三个 panel letters 使用统一的 figure-level 纵坐标，
保证 `(a)(b)(c)` 严格等高。Contour panel 使用外置 full-height colorbar，而不是
原 notebook 默认的 `46%` 高度 inset colorbar。
Contour 中保留起点圆圈和终点星号，但移除 `start`、`end` 文字标签。

该 PDF 实际出图时读取的 working run 与本 active folder 的 frozen copy 逐文件
SHA-256 完全一致。后续审阅和复现必须优先读取 active frozen path；working path
只保留原始来源谱系。

| 方法 | Active frozen data | 原始 working run | SHA-256 |
|---|---|---|---|
| IVM | `data/frozen_runs/tuned_ivm_alpha1e3_selected.npz` | `muller_q10_ivm/runs/inexact_tuned_resample/96b8903a356a.npz` | `3cf6649deb4b6bd2c11da51ca204a06513cad8e6702867e59619a2956d52162b` |
| ODE-NEB | `data/frozen_runs/ode_neb_k1000_selected.npz` | `muller_q10_ivm/runs/ode_default_production/ode_neb/ode_neb_17cd3c104e33_budget5000.npz` | `0c37cffbfbdb25a2388b48abf2aa7c18b2eb1178c450cf065fa5fd1641985c11` |
| Static NEB | `data/frozen_runs/static_neb_selected.npz` | `muller_q10_ivm/runs/refinement_production/static_neb/static_neb_9506398ecdd4_budget5000.npz` | `b1defffeaa72f0c62192cda812fbd5d6bb7787df19e13b342e8fed41213ec4dc` |
| Original String | `data/frozen_runs/original_string_selected.npz` | `muller_q10_ivm/runs/refinement_screen/original_string/original_string_cdb218376d2e_budget5000.npz` | `94a1c4c65dd7d98a2dbab65b236f61fa702662c31ce53560db0b81ef5a94b727` |
| OBSM-FIRE | `data/frozen_runs/obsm_fire_selected.npz` | `muller_q10_ivm/runs/refinement_production/obsm_fire/obsm_fire_463d17670289_budget4000.npz` | `82a9ac0f623d7f6f08f43f1fd69e3c4241e70936a447425d14b4be024c586546` |
| OBSM-LBFGS | `data/frozen_runs/obsm_lbfgs_selected.npz` | `muller_q10_ivm/runs/refinement_production/obsm_lbfgs/obsm_lbfgs_b36a738c93cc_budget5000.npz` | `82a7a74dbfbaa251ff9d514230f6e4313a57a571c91260faa9aa5936523fdc81` |

所有方法共享第 1 节定义的 fixed-Q、无 sin/cos 扰动、10D Muller-Brown 势能和同一条
22-image 初始路径。固定矩阵文件为 `data/inputs/Q_data.npz`，文件 SHA-256 为
`9a57ca4ce0aa39886aacbfa935c8c7db1423a7f7de3deb937697e72578415aca`；其中记录的
Q-array audit SHA-256 为
`372962a47504e6df28a20d8b022482a6835680765434b4385a6dba0e826974bc`。

本图使用的具体已选配置如下；完整 grid、选择理由和最终长程结果仍以第 2 节和第 4 节
为准。

| 方法 | 本图读取的选中配置 | 完整配置来源 |
|---|---|---|
| IVM | fixed `alpha=1e-3`; 9D `full_normal_lbfgs`; harmonic preconditioner; warm history 15; `new_call_cap=12`; accepted-step cap 1; cubic resample every 5 outer iterations; one pass; stationary at 2377 total forces | `data/reports/q10_inexact_tuned_resample_summary.json`, config `96b8903a356a` |
| ODE-NEB | ASE `NEBOptimizer(neb, method="ode")`; only `k=1000` tuned; improved tangent | `data/reports/baseline_final_selection.json`, `data/reports/q10_ode_default_k_sweep.json` |
| Static NEB | ASE `NEBOptimizer(neb, method="static", alpha=3e-4)`; `k=200`; improved tangent | `data/reports/baseline_final_selection.json` |
| Original String | `dt=3e-5`; Simplified-String Appendix-A cubic reparameterization every 2 iterations | `data/reports/baseline_final_selection.json` |
| OBSM-FIRE | `lambda=0.8`; 4 inner FIRE steps; `dt=0.01`; `beta=1`; cubic reparameterization every 5 iterations | `data/reports/baseline_final_selection.json` |
| OBSM-LBFGS | `lambda=0.2`; 2 inner L-BFGS steps; cubic reparameterization every iteration | `data/reports/baseline_final_selection.json` |

对应实验实现与选择代码如下。

| 职责 | 启用代码 |
|---|---|
| Q10 势能、力和 fixed Q 变换 | `code/common/q10_potential.py` |
| IVM proximal solver、improved tangent、history、preconditioner、cubic resample | `code/ivm/q10_ivm_core.py` |
| 最终 fixed-alpha tuned IVM grid | `code/ivm/run_q10_tuned_final_grid.py` |
| Original String、OBSM-FIRE、OBSM-LBFGS、Static NEB | `code/baselines/run_q10_baselines.py`, `code/baselines/run_q10_refinement.py` |
| ODE-NEB canonical ASE `k`-only sweep | `code/baselines/run_q10_ode_default_k_sweep.py` |
| 全路径 force accounting 审计 | `data/reports/q10_force_count_audit.json` |

对应绘图链如下。

| 职责 | 文件 |
|---|---|
| 本图唯一 active entry point；直接读取上述六个 frozen NPZ | `code/plotting/plot_q10_article_muller22_1x3_linear3000.py` |
| Active formal adapter；负责指标重建、正式样式和 Q10 contour | `code/plotting/plot_q10_article_muller22.py` |
| 本图冻结的正式 Muller-22 notebook 绘图源 | `code/plotting/reference/figure_M10_N20_v2.ipynb` |
| 输出 PDF | `figures/active_candidates/muller22_q10_article_method_comparison_1x3_linear3000.pdf` |

冻结 notebook 的 SHA-256 为
`f549e862ba701281ad19f258766dddd334e58b9cfb8ec0fe9f38ff159bd2c88f`。从本算例根目录
复现该图的命令为：

```bash
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  code/plotting/plot_q10_article_muller22_1x3_linear3000.py \
  --output figures/active_candidates/muller22_q10_article_method_comparison_1x3_linear3000.pdf
```

N10 当前 official 图只作为 out-of-scope legacy 复制到 `archive_legacy/`；不得拿来替代
本目录的 N20 结果。

## 7. 启用代码和数据索引

Baseline：

- `code/baselines/run_q10_baselines.py`：五类 baseline 核心实现及 broad grid。
- `code/baselines/run_q10_refinement.py`：targeted refinement。
- `code/baselines/run_q10_ode_default_k_sweep.py`：canonical ASE ODE k-only sweep。
- `code/common/q10_potential.py`：Q10 势能和力。

IVM：

- `code/ivm/q10_ivm_core.py`：9D exact/inexact proximal solver、history、preconditioner、resample。
- `code/ivm/run_q10_ivm_groups.py`：Groups 1-3。
- `code/ivm/run_q10_tuned_final_grid.py`：Group 4 locked grid。
- `code/ivm/tune_q10_inexact.py`：Group 4 形成过程的 tuning driver。

NN/连续表示：

- `code/nn_ivm/q10_nn_ivm.py`：PyTorch endpoint-exact MLP 基础实现。
- `code/nn_ivm/q10_nn_ivm_mlp_lbfgs.py`：Full MLP 与 RF L-BFGS 实验。
- `code/nn_ivm/q10_nn_ivm_pointwise.py`：早期 pointwise/RBF 实验。
- `code/nn_ivm/q10_nn_ivm_fourier.py`：Fourier/local/B-spline continuous projected IVM。
- `code/nn_ivm/run_q10_spline_joint_production.py`：最终三阶段 spline production。

关键机器报告：

- `data/reports/baseline_final_selection.json`。
- `data/reports/q10_force_count_audit.json`。
- `data/reports/q10_exact_no_resample_summary.json`。
- `data/reports/q10_exact_resample_summary.json`。
- `data/reports/q10_inexact_baseline_resample_summary.json`。
- `data/reports/q10_inexact_tuned_resample_summary.json`。
- `data/reports/q10_nn_ivm_dense_comparison.json`。
- `data/reports/q10_spline_joint_production_locked.json`。
- `data/reports/q10_mlp_random_feature_study.json`。

## 8. 复现入口

以下命令应从完整 working case 根目录执行，而不是直接在论文快照内执行：

```bash
cd /Users/wuzhiyou/Downloads/IVM_for_MEP

# Baseline broad/production and targeted refinement
python muller_q10_ivm/src/run_q10_baselines.py all
python muller_q10_ivm/src/run_q10_refinement.py

# Canonical ODE: ASE defaults, only k tuned
python muller_q10_ivm/src/run_q10_ode_default_k_sweep.py

# Four IVM groups
python muller_q10_ivm/src/run_q10_ivm_groups.py all
python muller_q10_ivm/src/run_q10_tuned_final_grid.py

# Locked cubic-spline continuous IVM, torch-nightly environment
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  muller_q10_ivm/src/run_q10_spline_joint_production.py
```

runner 的确切 CLI 参数以冻结源代码为准。不要在未审查的情况下把旧 tuning script
中的 candidate 路径当作 production。

## 9. 诚实限制和退出谱系的结论

- Original String 的 terminal barrier error 很小，但 `I_perp` 约 `5.9e2`，不能说两项都准。
- 22-image cubic densification 对收敛 MEP 存在约 `1.885e-3` 的传统 barrier floor。
- tuned discrete IVM 在离散 residual 上极强，但 5001 点 densified `I_perp` 约 509。
- 最终 spline 连续表示把 densified `I_perp` 降到约 4.14，但它在 4000 forces 才过主门槛；
  预先设定的 3000-force stretch gate 没有通过。
- Spline NN-IVM 不应被描述成 MLP；准确名称是 cubic-B-spline continuous IVM，项目文件仅沿用 NN-IVM 标签。
- Full MLP 与 RF 的结果明显不如 spline，不能只展示 transient checkpoint 作为 production。
- deterministic golden shift 使三个 nominal seed 结果完全一致，不能据此声称随机鲁棒性。
- 部分 baseline 的 barrier curve 有更低的孤立 transient minimum；当前支持的是同一 checkpoint 的联合优势。
- `main.tex` 当前 N20 PDF 与 working 最新 candidate 不同；在明确批准前不得静默覆盖。
- `alpha=1.25e-3`、旧 provisional NN 图、旧 pointwise-RBF production 和 N10 case 均退出当前启用谱系。

## 10. Archive 规则

`archive_legacy/` 中的材料只用于追溯，不得直接放回 `main.tex`。完整大规模 screen、
失败网络、临时 tuning 和所有原始 candidate 仍保留在 working `muller_q10_ivm/runs/`
与 `reports/`，因为移动它们会破坏已有 JSON/NPZ 的相对路径和复现链。论文侧不复制这些
大规模过时谱系，只保留说明、代表性 provisional 图和唯一源路径。

本次整理未修改 `paper/main.tex`，未覆盖 `paper/Figures/Muller10/`。
