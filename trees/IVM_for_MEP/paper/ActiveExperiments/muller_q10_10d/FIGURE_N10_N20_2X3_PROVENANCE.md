# Muller Q10 Nint=10/20 2x3 图：完整数据与代码溯源

状态：2026-08-20 启用 candidate 的专用 provenance。审查、复现、修改或引用下述
2x3 图前，先读本文件。若本文件的人工描述与 JSON、manifest 或冻结 NPZ 冲突，
以 `n12_final_selection.json`、`baseline_final_selection.json`、图 manifest 和冻结
NPZ 中的机器可读字段为准。

本登记不更新 `main.tex`，也不表示该图已经进入正式论文。

## 1. 图、plotter 和机器可读附件

| 角色 | 启用路径 |
|---|---|
| 2x3 PDF | `figures/active_candidates/muller_q10_article_method_comparison_n10_n20_2x3_linear3000.pdf` |
| 唯一绘图入口 | `code/plotting/plot_q10_article_n10_n20_2x3_linear3000.py` |
| 图中逐 checkpoint 数据 | `figures/active_candidates/muller_q10_article_method_comparison_n10_n20_2x3_linear3000_source.csv` |
| 完整 manifest | `figures/active_candidates/muller_q10_article_method_comparison_n10_n20_2x3_linear3000_manifest.json` |
| 正式风格 adapter | `image12/code/plotting/plot_q10_article_muller12.py` |
| 正式 notebook 快照 | `image12/code/plotting/reference/figure_M10_N20_v2.ipynb`，cell 7 |

`source.csv` 由 plotter 每次导图时重新生成，字段包括 case、method、
`number_of_forces`、`I_perp`、`Delta_E_dense` 和实际读取的冻结 NPZ 绝对路径。
CSV 保存冻结 run 中的全部 archived checkpoints；PDF 通过 xlim 只显示 0--3000。

`manifest.json` 同样由 plotter 生成，保存两排选中配置、selection report、
contour checkpoint、坐标轴口径、正式 notebook cell、全部冻结 NPZ 的绝对路径和
SHA-256。它是从 PDF 向底层数据追踪时最直接的机器可读入口。

## 2. 图的精确定义

- 上排 `(a--c)`：10 个可动 interior images，加 2 个固定端点，共 12 images。
- 下排 `(d--f)`：20 个可动 interior images，加 2 个固定端点，共 22 images。
- 第一列：离散路径上的 `I_perp`；使用正式 notebook 的指标代码。
- 第二列：500 点 coordinate-wise cubic densification 后的 energy barrier `Delta E`。
- 第三列：Muller 等高线及指定方法在共同 force window 内的路径。
- 横轴：线性 0--3000，标签为 `Number of Forces`。
- 上排隐藏 x-axis titles，仅下排承担共享的 `Number of Forces`/`z0` 说明；刻度仍保留。
- Barrier 纵轴：标准、未压缩的自动 log scale；不画原正式代码的 `1.1e2` 压缩阈值线。
- 曲线方法顺序：IVM、ODE-NEB、Static NEB、Original String、OBSM-FIRE、OBSM-LBFGS。
- IVM 在 run order 第一位并使用最高 zorder；颜色、线型和 legend 与启用 Muller 正式风格一致。

`Number of Forces` 不是 outer iterations，也不是单个 image 的 calls，更不是事后乘以
interior-image 数。它是所有可动 interior images 的 uncached force calls 的直接总和：

- 上排每个 checkpoint 对 10 个 interior-image ledgers 求和。
- 下排每个 checkpoint 对 20 个 interior-image ledgers 求和。
- 固定端点不计优化 cost；被拒绝的 trial force call 计费；cache hit 不重复计费。

## 3. Contour 的方法和 checkpoint

两排 contour 都只显示 IVM、ODE-NEB、OBSM-LBFGS。Plotter 不使用冻结 run 的
`images_seq[-1]`，而是按各自 `total_force_evaluations` 选最后一个
`Number of Forces <= 3000` 的 checkpoint，再把 `images_seq` 截断到该位置后调用
正式 notebook 的 `build_path_specs_from_runs`。

| 行 | IVM | ODE-NEB | OBSM-LBFGS |
|---|---:|---:|---:|
| `N_int=10` | 951 | 2970 | 2982 |
| `N_int=20` | 2377 | 2800 | 2737 |

这些数来自当前图 manifest 的 `contour_checkpoint_forces`，不是人工目测或四舍五入到
3000。由于 diagnostic interval 不同，各方法不一定恰好存在 3000-force checkpoint。

## 4. 共同物理体系和输入

两排均为固定 Q、无 sine/cosine 扰动的 10D Q-transformed Muller-Brown：

```text
z = Q @ x
U(x) = U_M(z_1, z_2) + 2 * sum_{j=3}^{10} z_j^2
grad_x U = Q.T @ grad_z U
force_x = -grad_x U
```

Q 由 seed 0 的高斯随机矩阵 QR 分解后冻结。启用输入为
`data/inputs/Q_data.npz` 和 `image12/data/inputs/Q_data.npz`；二者对应同一 Q，
SHA-256 为
`372962a47504e6df28a20d8b022482a6835680765434b4385a6dba0e826974bc`。

初始路径先在自然 z 坐标构造，再以 `x=Q.T@z` 变换。所有方法统一使用
Henkelman-Jonsson/ASE improved tangent。Original String 和两个 OBSM 使用
Simplified String Method Appendix 定义的自定义 cubic spline 重参数化。

## 5. 上排 Nint=10 的数据来源

### 5.1 唯一选择报告和选择规则

最终选择报告：`image12/data/reports/n12_final_selection.json`。

统一选择标准是在 0--3000 direct-total-force window 最小化等权
`AUC(log10(I_perp)) + AUC(log10(abs(Delta E_dense-Delta E_ref)))`，endpoint joint
score 只用于并列决胜。每种方法只与本方法的候选比较。完整 broad、production、
targeted refinement 和 alpha 报告均冻结在 `image12/data/reports/`。

Nint=10 实际完成 1674 个配置/复跑候选。本机运行 Original String、Static/ODE-NEB、
IVM 和最终 selector；远程 Mac mini 运行两个 OBSM，随后 rsync 回同一 working case
并由统一 selector 合并。完整网格范围、运行计数和诚实结果说明见
`image12/README_FIRST.md`。

### 5.2 图中选中配置

| 方法 | 图中配置 |
|---|---|
| IVM | fixed `alpha=2e-3`; full-normal L-BFGS; history 15; warm history; harmonic preconditioner; one accepted step/image/outer; max 12 new calls; cubic resample every 5 |
| ODE-NEB | ASE `NEBOptimizer(neb, method="ode")`; only `k=2000` tuned; `fmax=1e-14`; diagnostic 10 |
| Static NEB | ASE `NEBOptimizer(neb, method="static", alpha=5e-4)`; `k=0.05` |
| Original String | `dt=5e-4`; Appendix cubic reparameterization every 2 |
| OBSM-FIRE | `lambda=0.95`; inner 4; `fire_dt=0.01`; `beta=1`; Appendix cubic every 1 |
| OBSM-LBFGS | `lambda=0.25`; inner 2; Appendix cubic every 1 |

IVM 的 alpha 在单次 run 全程固定；没有 alpha schedule、mix 或 Lagrangian。其 image
subproblem 在完整 9D normal space 内无约束近似极小化
`U(phi_k + P_k delta) + ||delta||^2/(2 alpha)`。

### 5.3 图实际读取的冻结 NPZ

| 方法 | 冻结文件 |
|---|---|
| IVM | `image12/data/frozen_runs/tuned_ivm_alpha2e3_selected.npz` |
| ODE-NEB | `image12/data/frozen_runs/ode_neb_k2000_selected.npz` |
| Static NEB | `image12/data/frozen_runs/static_neb_selected.npz` |
| Original String | `image12/data/frozen_runs/original_string_selected.npz` |
| OBSM-FIRE | `image12/data/frozen_runs/obsm_fire_selected.npz` |
| OBSM-LBFGS | `image12/data/frozen_runs/obsm_lbfgs_selected.npz` |

### 5.4 Nint=10 实验代码

| 作用 | 启用代码 |
|---|---|
| 势能、Q、初始路径、指标基础 | `image12/code/experiments/q10_potential.py` |
| 五类 baseline 实现 | `image12/code/experiments/run_q10_baselines.py` |
| broad/production 分方法驱动 | `image12/code/experiments/run_q10_n12_baseline_search.py` |
| OBSM targeted refinement | `image12/code/experiments/run_q10_refinement.py` |
| ASE-default ODE k-only sweep | `image12/code/experiments/run_q10_ode_default_k_sweep.py` |
| Original/Static/ODE 最终局部网格 | `image12/code/experiments/run_q10_n12_final_baseline_refinement.py` |
| IVM 核心 | `image12/code/experiments/q10_ivm_core.py` |
| IVM 主 alpha grid | `image12/code/experiments/run_q10_tuned_final_grid.py` |
| IVM alpha 局部 refinement | `image12/code/experiments/run_q10_n12_ivm_alpha_refinement.py` |
| 统一 0--3000 selector | `image12/code/experiments/select_q10_n12_results.py` |

## 6. 下排 Nint=20 的数据来源

### 6.1 选择报告

- Baseline 选择：`data/reports/baseline_final_selection.json`。
- IVM fixed-alpha grid：`data/reports/q10_inexact_tuned_resample_summary.json`。
- ODE canonical k-only sweep：`data/reports/q10_ode_default_k_sweep.json`。
- 全部 Nint=20 实验范围和排除旧结果的规则：顶层 `README_FIRST.md`。

下排不是把 Nint=10 的最优超参数直接复制过去。Nint=20 参数来自原先独立完成并审计的
22-image 实验；组合图只负责在共同 0--3000 direct-total-force window 中并排展示。

### 6.2 图中选中配置

| 方法 | 图中配置 |
|---|---|
| IVM | fixed `alpha=1e-3`; 与上排同一 tuned full-normal L-BFGS policy |
| ODE-NEB | ASE `NEBOptimizer(neb, method="ode")`; only `k=1000` tuned; diagnostic 10 |
| Static NEB | ASE static; `k=200`; `alpha=3e-4`; diagnostic 10 |
| Original String | `dt=3e-5`; Appendix cubic every 2; diagnostic 25 |
| OBSM-FIRE | `lambda=0.8`; inner 4; `fire_dt=0.01`; `beta=1`; Appendix cubic every 5 |
| OBSM-LBFGS | `lambda=0.2`; inner 2; Appendix cubic every 1 |

### 6.3 图实际读取的冻结 NPZ

| 方法 | 冻结文件 |
|---|---|
| IVM | `data/frozen_runs/tuned_ivm_alpha1e3_selected.npz` |
| ODE-NEB | `data/frozen_runs/ode_neb_k1000_selected.npz` |
| Static NEB | `data/frozen_runs/static_neb_selected.npz` |
| Original String | `data/frozen_runs/original_string_selected.npz` |
| OBSM-FIRE | `data/frozen_runs/obsm_fire_selected.npz` |
| OBSM-LBFGS | `data/frozen_runs/obsm_lbfgs_selected.npz` |

### 6.4 Nint=20 实验代码

| 作用 | 启用代码 |
|---|---|
| 五类 baseline 实现 | `code/baselines/run_q10_baselines.py` |
| targeted baseline refinement | `code/baselines/run_q10_refinement.py` |
| ASE-default ODE k-only sweep | `code/baselines/run_q10_ode_default_k_sweep.py` |
| IVM 核心 | `code/ivm/q10_ivm_core.py` |
| tuned fixed-alpha IVM grid | `code/ivm/run_q10_tuned_final_grid.py` |

## 7. 一条命令重绘

从 `paper/ActiveExperiments/muller_q10_10d/` 执行：

```bash
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  code/plotting/plot_q10_article_n10_n20_2x3_linear3000.py \
  --output figures/active_candidates/muller_q10_article_method_comparison_n10_n20_2x3_linear3000.pdf
```

该命令同时覆盖 PDF、同名 `_source.csv` 和同名 `_manifest.json`。Plotter 不读取
working folder 的临时 runs，只读取上述论文侧冻结 NPZ 和 reports。

## 8. 审计顺序和诚实限制

1. 先读本文件确认图语义和 active paths。
2. 读同名 manifest 核对当前配置、contour checkpoint 和冻结 NPZ SHA-256。
3. 读同名 source CSV 核对 PDF 中任一可见 checkpoint。
4. 回到 Nint=10/Nint=20 selection reports 检查候选排序和选择标准。
5. 用冻结 NPZ 的 `config_json`、`total_force_evaluations`、`path_history`、
   `normal_force_history` 和 `delta_e` 做最终数据审计。

需要注意：第一列是离散 `I_perp`，第二列是 densified barrier，两者不是同一种
densification 证书。Contour 为最后一个不超过 3000 的 archived checkpoint，不保证
恰好等于 3000。Nint=10 与 Nint=20 的最优 alpha 和 baseline 超参数是各自独立调参所得，
不应把两排的差异解释为只改变 image 数、其他超参数完全相同的 controlled ablation。
