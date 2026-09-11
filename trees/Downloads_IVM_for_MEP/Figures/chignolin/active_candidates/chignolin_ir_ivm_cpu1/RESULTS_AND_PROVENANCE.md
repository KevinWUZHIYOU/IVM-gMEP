# Chignolin IR-IVM 迁移实验：结果、审计与溯源

> 这是论文候选图目录中的同步溯源副本。本文所有相对路径均以权威实验根目录
> `/Users/wuzhiyou/IVM_for_MEP/python/chignolin_ir_ivm_20260826` 为基准；同名原始文档也保存在该目录。

本文件记录 Ala IR-IVM 向 Chignolin 的最小必要迁移、三参数调节、正式结果和完整数据链。所有结论均来自本目录中的预注册、原始优化结果、逐结果审计、加密势垒诊断及作图溯源文件；未把未提交的物理步、recovery 中间 trial 或终止时未闭合的尾段当成可报告路径状态。

当前 `chignolin_ir_ivm_overlay_total_nf.pdf` 的单图自包含登记位于本目录的 `chignolin_ir_ivm_overlay_total_nf_provenance.md`。本文件保留整个参数实验和 recovery 审计；判断该图的具体数据输入时优先使用单图登记及 same-stem 机器附件。

## 1. 主要结论

Chignolin 上正式选出的 IR-IVM 参数为

\[
\alpha=10,\qquad r=0.5,\qquad m_{\max}=5.
\]

这里的 \(m_{\max}\) 是“已计算整条路径的离散内部步数上限”，不是原子位移上限；物理原子位移上限始终固定为 \(1\ \text{Å}\)。按 2026-08-27 的候选图决定，这套算法及其数据在图例中直接记为 `IVM`，旧 V9 IVM 不再叠加。正式单线程结果在同一个已经提交重参数化的路径状态、总 Number of Forces 为 905 时达到

\[
I_\perp=0.1852701724454184
\le 0.21242215133173672,
\]

统一弧长 500 点定义下，同一状态还达到

\[
\Delta E_{500}=24.097582386424108\ \text{kcal mol}^{-1}
\le 24.17608142601358\ \text{kcal mol}^{-1}.
\]

新图对每条入图路径都在未加权 \(3N\) 笛卡尔空间中按累计弧长取 500 个总点（含两个端点），并使用同一 OpenMM CPU/double 模型计算 \(\Delta E_{500}\)。共同阈值下的首次达到结果为：

- \(I_\perp\) 面板：IVM 为 905，ODE-NEB 为 17662，OBSM-FIRE 为 19942；
- \(\Delta E_{500}\) 面板：IVM 为 905，ODE-NEB 为 15842，OBSM-FIRE 为 19942。

图中的倍率箭头按用户指定只比较 IVM 与 ODE-NEB：左图为 \(19.5\times\)，右图为 \(17.5\times\)；OBSM-FIRE 的结果仍保留为曲线和阈值定义证据，但不画倍率箭头。旧 overlay 中的 1502 和 \(1.66\times\) 来自旧 IVM 势垒历史的稀疏时间取样，已被本次替换图废止；新图既不显示旧 V9 IVM，也不再给出“相对旧 IVM”的箭头。由于预注册的三参数调节已经取得严格改进，没有启用第四个算法技巧。

## 2. IR-IVM 的算法身份：重参数化是强制组成部分

本实验中的 IR-IVM 不是“可选重参数化”的算法。每次外层迭代都必须进入 `state_curve` 重参数化及其 recovery 事务；跳过这一过程的运行不属于这里报告的 IR-IVM。一次外层更新按以下逻辑完成：

1. 求解联合的 inexact 离散法向近端子问题；
2. 使用固定的 one-way FFrbk/L-BFGS selector 选择内部步，选中步的乘子为 `eta=1.1`，历史长度为 7；
3. 对更新后的整条路径执行 `state_curve` direct reparameterization proposal；
4. 若 direct proposal 未满足逐图像接受判据，则进入从 Ala 原样迁移的 recovery 事务，并用原有双判据合并规则决定是否提交；
5. 只有初始路径或 `reparam_committed=True` 的完整路径才可进入首次达标判断、加密势垒计算和文章图。

固定不变的算法条件还包括：`reparam_every=1`、逐图像 direct acceptance、`I_perp` monotone outer globalization、`discrete_b040` 子问题、选中主步不使用 energy fallback、物理最大原子位移为 \(1\ \text{Å}\)。本实验只允许改变 `alpha`、相对残差阈值 `r` 和内部整带步数上限 `m_max`。

正式 5000 内部 NoF 结果共记录 13 次外层重参数化尝试，其中 11 次提交；文章图使用初始路径和这 11 个已提交路径，共 12 个状态。文件中 `reparam_attempted=True` 的 265 条记录是 direct/recovery trial 的历史记录数，不是 265 次独立外层重参数化尝试。未提交的中间状态全部保留在成本账本中，但不进入科学指标曲线。

## 3. Chignolin 基线与体系

### 3.1 文章基线

| 项目 | 只读来源 | SHA256 |
|---|---|---|
| 文章 V9 比较图 | `/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/chignolin/chignolin_cost_to_precision_total_nf_v9_20260730.pdf` | `c56e61e7575b7f1cc784b946bd37e65682204579f7f85c3115b200b2dbdf37ff` |
| 当前 Chignolin V9 IVM 原始结果 | `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/runs/ivm/ivm_fixed_alpha_v9_kkt0p7_cap30_a1_central_state_ivm_cfg9b5a238786a3_budget650.npz` | `04a29ae267b3c6b953dac25e33db462a27d80574cc5510feb070484ea02f8691` |
| V9 方法比较汇总 | `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/figures/article_update_20260730_v9_final/chignolin_method_comparison_summary.json` | `7d6c593ac70fe311d345364435e38d2941d494172c2dc70acfbfac922cb33754` |

原始图和 V9 数据只读，未被覆盖。基线、所有比较方法及阈值的固定路径和哈希登记在 `registrations/FIGURE_COMPARISON_CONTRACT.json`。

### 3.2 共同初始路径和能量模型

- 路径共有 22 个图像，其中 20 个为内部图像；每个图像含 166 个原子，展平路径形状为 `(22, 498)`。
- 共同初始路径数组 SHA256 为 `57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996`。
- 本地初始路径容器 `inputs/common_initial_path_equidistant.npz` 的文件 SHA256 为 `18849a069d57f07ec1beb26eb79ca29c6b23d8881795e0d6a4ca97ebf9c6eba2`。
- 拓扑 `inputs/chignolin_clean.pdb` 的 SHA256 为 `04b122a0b98492b6fadae35772ffc307b8916ea65c8eaf959cb5187bbbfeab01`。
- OpenMM 使用 `amber14-all.xml` 和 `implicit/gbn2.xml`，`NoCutoff`、CPU、double precision。

## 4. 参数实验及选择

### 4.1 已完成的 37 组参数设置

本次调节不是把 240 种三维组合全部机械展开，而是按预注册规则先定位 `alpha`，再在选中的 `alpha=10` 上完整展开 \(r\times m_{\max}\)。去除阶段之间的重复设置后共有 37 组：

| 阶段 | 固定量 | 变化量 | 独立设置数 | 内部 NoF 预算 |
|---|---|---|---:|---:|
| `alpha` 扫描 | `r=0.5, m_max=5` | `0.1, 1, 10, ..., 10^6` | 8 | 1500 |
| 完整 `r × m_max` 扫描 | `alpha=10` | `r={0.1,0.3,0.5,0.7,0.8}`，`m_max={5,8,10,15,20,50}` | 30 | 已在顺序扫描中的设置为 1500，其余交互设置为 1000 |
| 去重总数 | — | `8 + 30 - 1` | **37** | — |

在冻结的 \(I_\perp\le0.21242215133173672\) 判据下，37 组设置中 8 组达到阈值，29 组在最后一个完整重参数化路径处仍未达到，0 组审计失败。未达到的设置按最后一个完整路径状态作右删失记录；该 NoF 只是已经观测到的下界，绝不被解释为首次达标成本。

达到阈值的 8 组设置如下。该表是两线程筛选阶段的结果，用于选参；正式报告值见第 6 节的单线程复现。

| `alpha` | `r` | `m_max` | 筛选预算 | 首次达到 \(I_\perp\) 的 NoF |
|---:|---:|---:|---:|---:|
| 10 | 0.5 | 5 | 1500 | **907** |
| 10 | 0.3 | 8 | 1000 | 936 |
| 10 | 0.3 | 10 | 1000 | 961 |
| 10 | 0.3 | 20 | 1000 | 961 |
| 10 | 0.5 | 50 | 1500 | 1170 |
| 10 | 0.5 | 8 | 1500 | 1218 |
| 10 | 0.8 | 5 | 1500 | 1273 |
| 10 | 0.7 | 5 | 1500 | 1396 |

37 组设置的逐行参数、首次达标、右删失、结果路径、结果 SHA256 和 configuration SHA256 均在 `reports/parameter_scan_combined/parameter_settings.csv`。长表作图数据在 `parameter_effects_long.csv`，完整机器可读分析在 `parameter_scan_analysis.json`。

### 4.2 从控制变量实验得到的规律

- `alpha` 不是越大越好。在 `r=0.5, m_max=5` 下，只有 `alpha=10` 在筛选预算内达到阈值；`0.1, 1, 10^2, ..., 10^6` 均未达到。
- 在 `alpha=10, m_max=5` 下，中间的 `r=0.5` 最快；更严格的 `r=0.1,0.3` 在 1500 预算内未达到，而 `r=0.7,0.8` 虽能达到，成本高于 `r=0.5`。
- 在 `alpha=10, r=0.5` 下，`m_max=5` 最快；增大内部步数并不单调改善结果。`m_max=8` 和 50 分别在 1218 和 1170 达到，10、15、20 在 1500 预算内未达到。
- 在额外的交互设置中，`r=0.3` 配合 `m_max=8,10,20` 能在 1000 附近达到，但都没有超过 `(10,0.5,5)` 的 907。

因此，适用于这类实验的稳健调参顺序是：先按十倍尺度扫描 `alpha`；再在最佳数量级附近扫描中间强度的 `r`；最后从较小 `m_max` 起逐步增加。每一步都应以“已提交重参数化路径上的首次达标 NoF”为选择量，并在候选进入正式比较前补算同一状态的加密 \(\Delta E\)。这个顺序是本实验给出的经验方法，不构成对所有体系的单调性假设。

### 4.3 5000 内部 NoF 延长实验

两线程筛选后，三组候选被延长到 5000 内部 NoF，并对所有已提交路径补算加密势垒：

| `alpha, r, m_max` | \(I_\perp\) 首次达标 NoF | 两指标联合首次达标 NoF |
|---|---:|---:|
| `10, 0.5, 5` | 966 | **966** |
| `10, 0.3, 8` | 936 | 1225 |
| `10, 0.3, 20` | 1067 | 1361 |

这一步说明仅按 \(I_\perp\) 选取 `(10,0.3,8)` 会遗漏加密势垒成本；联合判据仍选择 `(10,0.5,5)`。

## 5. 为什么正式值采用 `OPENMM_CPU_THREADS=1`

最初参数筛选使用两个 OpenMM CPU 线程。检查发现，两线程在第一次优化步之前就产生了非 bitwise-identical 的力，随后放大为一次 recovery 接受分支差异；同一 `(10,0.5,5)` 设置在 1500 筛选中首次达到 \(I_\perp\) 的 NoF 为 907，而独立的 5000 延长结果为 966。这不是预算裁切造成的差异，而是浮点归约次序对离散接受分支的影响。

当前文章 V9 Chignolin 基线本身使用一个 OpenMM CPU 线程。为与基线公平对齐并检验可复现性，在首次单线程 OpenMM 运行之前登记了 `registrations/CPU1_PREFIX_REPRO_PREREGISTRATION.json`。单线程适配器只把 `OPENMM_CPU_THREADS` 从 2 改为 1；IR-IVM 的算法、参数、强制 `state_curve` 重参数化和 recovery 均未改变。

随后从同一初始路径独立运行 1500 和 5000 两个内部预算。严格前缀审计结果为：

- 两次运行的初始路径 bitwise identical；
- 所有算法控制量相同；
- 截至短运行最后一个已提交路径（总 NoF 1455），70 个逐记录科学数组逐位一致；
- 唯一排除的是非科学量 `wall_seconds`；
- 不一致数组数为 0；
- 1500 和 5000 运行都在总 NoF 905 首次达到 \(I_\perp\) 阈值。

因此正式比较采用单线程 5000 结果。`OPENMM_CPU_THREADS=1` 是执行可复现性设置，不是新的优化技巧。

## 6. 历史选参阶段的 adaptive-dense 联合检查（不用于当前图）

**本节保留的是在 arc-length 500 协议冻结之前使用的旧 Cartesian 分段 adaptive-dense 检查，只用于说明当时为什么选中 `(10,0.5,5)`。本节的 `24.100098...`、5030 次诊断和旧 V9 IVM=1502 均不是当前候选图右面板的数据，不得据此复画当前图。** 当前图统一使用第 8.5、9.4 节登记的 arc-length 500 数据：IVM 在 NoF 905 时为 `24.097582386424108 kcal/mol`，共同阈值为 `24.17608142601358 kcal/mol`，且旧 V9 IVM 不入图。

正式单线程结果中，可用完整路径记录的总 NoF 为

`22, 126, 210, 386, 567, 735, 905, 1220, 1455, 1697, 1947, 3740`。

在该历史 adaptive-dense 协议下，NoF 905 的已提交路径同时满足当时的两项冻结判据：

| 方法 | \(I_\perp\) 阈值首次达到 NoF | 加密 \(\Delta E\) 阈值首次达到 NoF | 联合首次达到 NoF |
|---|---:|---:|---:|
| IR-IVM `(10,0.5,5)` | **905** | **905** | **905** |
| 当前 V9 IVM | 1382 | 1502 | 1502 |

联合达标必须发生在同一个已提交的路径记录，不能把两个不同记录的最佳值拼接。正式 IR-IVM 在 NoF 905 处的两项实际值分别为 `0.1852701724454184` 和 `24.100098154200122 kcal/mol`。

加密势垒遵循与文章基线相同的 Cartesian 分段协议：每段先用 10 和 20 个子区间比较；若相邻分辨率的势垒差超过 `0.05 kcal/mol`，再提高到 40 和 80。正式 12 个路径状态均在 20 个子区间处通过收敛判断。诊断过程产生的 5030 次能量计算完全排除在优化 NoF 横轴之外。

## 7. Number of Forces 的定义与 recovery 审计

### 7.1 成本口径

一次真实的联合能量/力 cache miss 对一个图像计为一次 NoF。每条历史记录的文章成本为

\[
N_F=\sum_{i=1}^{22}\texttt{counter\_vectors}[i].
\]

该定义包括：

- 20 个内部图像的所有已接受和未接受内部步；
- direct reparameterization proposal；
- recovery 中的所有中间 trial 和回退路径；
- 两个固定端点各一次初始计算。

因此总 NoF 始终等于内部 NoF 加 2。正式首次联合达标的 905 等于 903 个内部 NoF 加两个端点；5000 内部预算的最终账本为 5002。单独的加密势垒诊断不进入该账本。

正式 5000 结果的 companion audit 给出

`backend_raw_joint_evaluations = sum_image_joint_evaluations = 5002`，

且 22 个逐图像计数与结果 NPZ 的最后一个 `counter_vectors` 完全一致。最终一次未完成事务形成 1262 个内部 NoF 的尾段；这些计算已收费，但由于没有形成新的已提交重参数化路径，最后一个可用于科学比较的路径仍是 NoF 3740。

### 7.2 recovery 的 399 次“未归入保存记录”不是漏算

正式长运行的 recovery 标量总计为 1943 次联合计算。逐个保存记录可直接归属 1544 次，另有 399 次无法归到一个保存的历史闭合记录。原因是一次完整 rollback recovery 的闭合记录被 `calls_per_image` 去重：397 次中间调用及 2 次闭合调用都进入了底层逐图像计数，但没有单独的保存记录承接整个事务。

更新后的审计器只在以下证据同时成立时允许这种记录归属差额：recovery 总标量不小于逐记录之和；显式闭合加缺失闭合与重参数化尝试数相容；差额位于逐图像计数给出的边界内；终止状态不存在被误当作完整路径的开放事务。该结果中：

- recovery 标量总数：1943；
- 已归入保存记录：1544；
- 未归入保存记录：399；
- 缺少的闭合记录：1；
- 逐图像计数给出的允许区间：397 至 416；
- 399 位于该区间内；
- 最终总账仍严格满足 `5002 = 5002`。

所以 399 表示“无法归属到某一条保存记录”，不是“没有计入 NoF”。所有 43 个优化结果经过最终统一复审均通过，总终止 NoF 为 67059；其中只有这一个单线程长结果存在非零的保存记录归属差额。

## 8. 文件与 SHA256 溯源

### 8.1 预注册和比较契约

| 文件 | SHA256 | 作用 |
|---|---|---|
| `registrations/CAMPAIGN_PREREGISTRATION.json` | `46c2649d693deae21305971434ac6c010d048405a5ee167965064536cbca7647` | 首次 OpenMM 运行前冻结算法、参数范围、成本和选择规则 |
| `registrations/CPU1_PREFIX_REPRO_PREREGISTRATION.json` | `2e6207db69b802a6651dab943a9ef21170d6019360f42bbb5218dc5ccdc83444` | 首次单线程运行前冻结唯一执行改动及前缀检验 |
| `registrations/FIGURE_COMPARISON_CONTRACT.json` | `fe28509d0d0ca8bee093b9b8cd87aef9d5af4291c7d7bc794925734bcd13829d` | 固定 V9 基线文件、哈希、阈值及作图门禁 |
| `registrations/screen/*.json` | 每组文件内自带 source/input SHA256 和完整命令 | 37 组筛选的逐运行登记 |
| `registrations/production/*.json` | 每组文件内自带 source/input SHA256 和完整命令 | 三组 5000 延长实验登记 |
| `registrations/repro_cpu1/chig_irivm_a1e1_r05_m05_inf1500_cpu1_prefix.json` | `2c0c03c63d61e8956f9239e41476e07e29dff026275a0e2b77e6ab2b23aa0900` | 单线程 1500 前缀运行登记 |
| `registrations/repro_cpu1/chig_irivm_a1e1_r05_m05_inf5000_cpu1_prefix.json` | `1cbe875c828b1a0591220fc9902ed60d332b2a631f4a476124187e766e87e16f` | 单线程 5000 正式运行登记 |

### 8.2 迁移后的算法代码

| 文件 | SHA256 |
|---|---|
| `src/chignolin_core.py` | `ad5d6532f7acd7c7b2560fdb68e6ead796374bcf0b24d1c0790b7e51fa5e4faa` |
| `src/run_ivm_v9_spring_projected_production_20260730.py` | `5a9a33625867b19200f20fca9675c3b3a3e1a25ca428b1e149397c7530474bd5` |
| `src/frozen_discrete_b040.py` | `a42da93b10ab9e235770569159cfaf5bd36e819ed169ccb619729b5de6df9ed4` |
| `src/run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py` | `75f4dc5f4d7ab6c7af13e5b8e15a0f81b41b4a937346233a20a3e20e0de160f4` |
| `src/run_ivm_v9_discrete_selected_eta_shim.py` | `a2f071ea787a008ed001fd806b8952f84d8db90335b544308478f3f6ae2facf9` |
| `src/run_ivm_v9_discrete_ffrbk_takeover_grid_shim.py` | `4ee5cb4ad5536c7785dc6d017402af230f2451e1644cf7ef53d9de098757a6ae` |
| `src/run_ivm_v9_total_internal_nf_budget.py` | `ca0fa9d44334094f406be6e3308b53326d9b81c18cc839feed831a184e764e58` |
| `src/run_chignolin_ir_ivm_member.py` | `d68f2db9857eb9b0f9d3bcfd2ba99261e34a216f81a0ee053d3d3e68030fee05` |
| `src/run_chignolin_ir_ivm_batch.py` | `3a8603e009c52c3affe3b04b8d34aed8bd1ccc468df010f1fd5a93c062669f7d` |
| `src/run_chignolin_ir_ivm_cpu1_repro.py` | `b89bc82f13074ed3a81db2053ce7641291ae5d1e16ab49fab2ae57a7d9a3e779` |

前七个模块构成从 Ala 迁移并冻结的 IR-IVM 核心；`member` 和 `batch` 只负责 Chignolin 输入、预登记、预算与顺序执行，`cpu1_repro` 只把 OpenMM CPU 线程数固定为 1。每台主机使用 `logs/chignolin_ir_ivm_host.lock` 的 host-local `fcntl` 锁，批处理顺序运行，避免同一主机同时启动两个 Chignolin 优化任务。

### 8.3 原始结果、审计与分析

| 文件 | SHA256 | 作用 |
|---|---|---|
| `reports/final_all_optimization_audit.csv` | `d53a6bb949dc325516b9508df75f2cc5ff0c565740626a8df01dbfbe95e59f16` | 43 个优化结果的逐行最终审计 |
| `reports/final_all_optimization_audit.json` | `75d60efadbac4feb325516d9baaea5ec7e53492880ad0010840d9d10cfd9b7f8` | 43 个结果的完整路径、结果 SHA 和 recovery 证据 |
| `reports/parameter_scan_combined/parameter_settings.csv` | `1e65a0ebb30bc539c86efd028723255fb4c8bb2b4a4d728a1316676ae49e35fe` | 37 组参数设置的唯一汇总和每组结果 SHA |
| `reports/parameter_scan_combined/parameter_effects_long.csv` | `5a8e94177305245e35117586783482100824b124c1486b4473206c98806df40d` | 三个参数轴的长表数据 |
| `reports/parameter_scan_combined/parameter_scan_analysis.json` | `7ebb284be8126084666f868db4b87789e57ff3a1eb5a24a79d85d46ed6d31684` | 37 组机器可读分析、审计来源及右删失解释 |
| `reports/cpu1_prefix_repro/prefix_comparison.json` | `00038326686746c7fae3cbca2c97cbf2858d52b0f1c4d23ceae0f88f0a5e8b1a` | 70 个记录数组严格前缀检验 |
| `reports/cpu1_prefix_repro/audit_summary.json` | `f6abfae1e591461d860162942650d28a7f8b993a3a6173e610062a117fdaf354` | 两个单线程结果的统一审计 |
| `reports/cpu1_prefix_repro/audit_1500.json` | `eff6369c2ac00eea9b4070132b9a4f18a72925d69745ea3d05fed60657100f58` | 参数影响图中的单线程 905 确认点 |

43 个原始优化结果不在本文件逐项重复，以避免手工清单与结果目录分叉。其完整路径、SHA256、算法参数、首次达标、终止成本和 companion audit SHA 均以 `reports/final_all_optimization_audit.json` 为权威清单；37 组参数设置还可由 `parameter_settings.csv` 反向定位。目录关系为：

- `runs/smoke/`：1 个完整流程与计数冒烟实验；
- `runs/screen/`：37 个参数筛选结果；
- `runs/production/`：3 个两线程 5000 延长结果；
- `runs/repro_cpu1/`：2 个单线程严格前缀结果；
- 每个优化 NPZ 旁边的 `.audit.json`：逐图像底层调用计数审计。

正式短、长结果及关键诊断为：

| 文件 | SHA256 |
|---|---|
| `runs/repro_cpu1/ivm_chig_irivm_a1e1_r05_m05_inf1500_cpu1_prefix_central_state_ivm_cfg29d8ce2d7afa_budget1500.npz` | `196695a8fc2def89b369ab65fba5f9717e4c36457f8e92c09be5f830a7f10fa2` |
| 上述短结果的 `.audit.json` | `343d694b3c36595d9ac7f53ddd6c8479942fbf0004e82678f6655a4c440c59f3` |
| `runs/repro_cpu1/ivm_chig_irivm_a1e1_r05_m05_inf5000_cpu1_prefix_central_state_ivm_cfg844e61a3b6af_budget5000.npz` | `e8ce449e55b975360bbc29fd49a8e28c1fe8c0618382eb37b0a86e022bdfe03a` |
| 上述正式长结果的 `.audit.json` | `f6ae6899ef8cdf3ad8c09adcd6aad2891cec91a6ddc7a52e6d8f03414a314085` |
| `runs/diagnostics/dense_cpu1_a10_r05_m05_all_committed.npz` | `af1d8365ddc25cc4372eac5c53ff48facace9ce94193f20229b181c948214d8a` |
| `runs/diagnostics/dense_cpu1_a10_r05_m05_all_committed.audit.json` | `17781c5b7ab374124ec035b0adf5f32325dfa2e64ae6c40d8d08cfeb544078ab` |
| `runs/diagnostics/chignolin_arc500_independent_raw.npz` | `15e4ed61e0e9c213055060566b98ee4739cc47263c2b5ab53f881153fe3a81f1` |
| `runs/diagnostics/chignolin_arc500_independent_raw.json` | `c0adb3aa0f37692abd719f90be7afa8de6e1c88a007361885e2f4e33acddc1b7` |
| `runs/diagnostics/chignolin_ivm_arc_length_500.npz` | `3def8ffb3c70342d61285813511203c2970e30cb0146f37aa1d99aa3ce20a564` |
| `runs/diagnostics/chignolin_ivm_arc_length_500.audit.json` | `03bca53e4390fd2a0373901ef7cea3897ffb4ff69717418924306defd17cd509` |

### 8.4 分析与作图代码

| 文件 | SHA256 |
|---|---|
| `src/audit_chignolin_ir_ivm_results.py` | `11e91843e464f156b00f9c6a21374dd5f30a3db4080537c03d41cbd761d5ef88` |
| `src/analyze_chignolin_ir_ivm_parameter_scan.py` | `4189786b63abd80f259b9d721390b8250fd80ba2cc741190161c5b1e248c5c50` |
| `src/compare_cpu1_prefix.py` | `2d6db5b3c7ed70676ea8123b06779e7cad7f8dcdf0f359828107f17d7abff61a` |
| `src/postprocess_irivm_dense_by_record.py` | `2b49316f642fb8e4d714b6df7e970d1d701957738c2ee04a3b7eb401270fbb64` |
| `src/plot_chignolin_irivm_vs_v9.py` | `494b6d7569b4695d9799cdcbaf8400e0d30e641f0750fc7e9a438670b5f2d38f` |
| `src/plot_chignolin_irivm_parameter_effects.py` | `4b634ab0c55f3a909fa97d6851383e4a70ec9cebc188dd3b95dfc96de20a541a` |
| `src/postprocess_arc_length_500.py` | `607c0ceb41f3de4f6cf40d8094f37947bb687f72d4c516be219453cb5bbe4a17` |
| `src/plot_chignolin_ivm_arc500.py` | `62d30b5b81e0ab3b42bdf080308e95c615a4ecee0b960946b1df4dd15e0743b9` |
| `src/plot_chignolin_ivm_three_metrics.py` | `689e1a4322305a4e1b2d79ee9b557f2fa20625d6ac648550e91b9ca15c18bf73` |
| `tests/test_arc_length_500.py` | `dc387d3b6e9638b76182493119f43e83afaedc2db075b2141fd17569722a4cd2` |
| `registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json` | `f351020439a1ef7d75bd0c21ae6d843722d813912542bf5cef984f3f0ab424b8` |

### 8.5 已登记并晋升到论文的图

新的 IVM 替换图位于论文工作树的 `Figures/chignolin/active_candidates/chignolin_ir_ivm_cpu1/`；参数影响图仍位于实验根目录。2026-09-06，候选 PDF 的逐字节相同副本晋升为 `/Users/wuzhiyou/Downloads/IVM_for_MEP/paper0819/Figures/chignolin/chignolin_ir_ivm_overlay_total_nf.pdf`，并由 `/Users/wuzhiyou/Downloads/IVM_for_MEP/paper0819/main.tex` 引用；纸稿本地的 `*_SOURCE.md`、frontier CSV、annotations CSV 和 provenance JSON 构成同目录溯源包。可复验的论文预览固定输出到 `paper0819/tmp/vscode-latex/main.pdf`，根目录 `main.pdf` 不作为本次登记交付物。图形版式以 `/Users/wuzhiyou/IVM_for_MEP/output/pdf/alanine66d_delta_aware_grid_ivm_replacement_1x2_preview.pdf`（SHA256 `483530e8c0d764e2c4b4c82af3c280cc7f8fee300eb35b12db6e1260759f3a44`）为样式参照，并按要求在每幅图中只画 IVM 相对 ODE-NEB 的一根倍率箭头。

| 文件 | SHA256 | 数据来源 |
|---|---|---|
| `chignolin_ir_ivm_overlay_total_nf.pdf` | `06f402095523217f04ea2847e2e9b7a9d1f7d141fa807393be68b3f04c709396` | 三个正方形子图：I_perp、完整绘制后按原范围裁切的弧长 500 点 Delta E、以 `S_hat` 显示的离散 geometric action；仅以箭头比较 ODE-NEB |
| `chignolin_ir_ivm_overlay_total_nf.png` | `3a88ebf4a8563cf3084569cf3150b13b38ca76f96affa03657c494b7a38a7d45` | 同上 |
| `chignolin_ir_ivm_overlay_total_nf.svg` | `30256a759252ccbf930508d24fa72a07618e668502adb40e665727fd1dff7ce4` | 同上 |
| `chignolin_ir_ivm_overlay_total_nf_frontier.csv` | `3705876f7a14861b242d6f65dfa027c2e30acb9d67e48840b0f16d945294b6ab` | 五种方法、三个指标的完整精度—成本前沿 |
| `chignolin_ir_ivm_overlay_total_nf_annotations.csv` | `d714baaea66997aeeefb2b768e6ca846fc99b40518d79cec0501b1a71ce3fe6c` | 三个面板的阈值、IVM—ODE-NEB 首次达到 NoF 和倍率 |
| `chignolin_ir_ivm_overlay_total_nf_provenance.json` | `bc56c16d94a950da4c4d618e39558e3dff8d62144cd88bf94633c81c305a291b` | 两类诊断、三指标数据映射、Delta E 完整绘制后裁切、正方形版式和首次达到验证 |
| `chignolin_ir_ivm_parameter_effects.png` | `c36815c6ea3d8deb0d0e606f86a92d93b5bea5d84f833544b6745f8ffb5982e0` | 37 组参数设置及 CPU1 正式确认点 |
| `chignolin_ir_ivm_parameter_effects.svg` | `f0050e4115aef88d5e993535a08bd3eeb5398edbdb2c2a0620b3626942948d25` | 同上 |

参数影响图中未达到阈值的设置只以 `right_censoring_total_nf == last_complete_reparam_total_nf` 的右删失下界显示；没有把删失值画成首次达标值。

## 9. 复现命令

以下命令使用本次运行时解释器。现有结果和登记文件带有防覆盖门禁；要重新从头运行，应在保留同一输入与源码哈希的干净实验副本中执行，或先使用只读的 `--dry-run`/审计命令。

```bash
CHIG_IR_ROOT=/Users/wuzhiyou/IVM_for_MEP/python/chignolin_ir_ivm_20260826
CHIG_IR_PY=/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python
cd "$CHIG_IR_ROOT"
```

### 9.1 参数扫描

```bash
"$CHIG_IR_PY" src/run_chignolin_ir_ivm_batch.py \
  --root "$CHIG_IR_ROOT" --name alpha_screen_r05_m05_inf1500 \
  --tier screen --internal-nf-budget 1500 \
  --settings 0.1,0.5,5 1,0.5,5 10,0.5,5 100,0.5,5 \
             1000,0.5,5 10000,0.5,5 100000,0.5,5 1000000,0.5,5

"$CHIG_IR_PY" src/run_chignolin_ir_ivm_batch.py \
  --root "$CHIG_IR_ROOT" --name r_screen_a10_m05_inf1500 \
  --tier screen --internal-nf-budget 1500 \
  --settings 10,0.1,5 10,0.3,5 10,0.5,5 10,0.7,5 10,0.8,5

"$CHIG_IR_PY" src/run_chignolin_ir_ivm_batch.py \
  --root "$CHIG_IR_ROOT" --name mmax_screen_a10_r05_inf1500 \
  --tier screen --internal-nf-budget 1500 \
  --settings 10,0.5,5 10,0.5,8 10,0.5,10 10,0.5,15 10,0.5,20 10,0.5,50

"$CHIG_IR_PY" src/run_chignolin_ir_ivm_batch.py \
  --root "$CHIG_IR_ROOT" --name interaction_screen_a10_inf1000 \
  --tier screen --internal-nf-budget 1000 \
  --settings 10,0.1,8 10,0.1,10 10,0.1,15 10,0.1,20 10,0.1,50 \
             10,0.3,8 10,0.3,10 10,0.3,15 10,0.3,20 10,0.3,50 \
             10,0.7,8 10,0.7,10 10,0.7,15 10,0.7,20 10,0.7,50 \
             10,0.8,8 10,0.8,10 10,0.8,15 10,0.8,20 10,0.8,50
```

各设置在 OpenMM 启动前写入 `registrations/screen/`；批处理持有 host-local `fcntl` 锁且按顺序运行。

三组候选的 5000 延长命令为：

```bash
"$CHIG_IR_PY" src/run_chignolin_ir_ivm_batch.py \
  --root "$CHIG_IR_ROOT" --name production_top3_inf5000 \
  --tier production --internal-nf-budget 5000 \
  --settings 10,0.5,5 10,0.3,8 10,0.3,20
```

### 9.2 单线程严格前缀复现

```bash
"$CHIG_IR_PY" src/run_chignolin_ir_ivm_cpu1_repro.py \
  --root "$CHIG_IR_ROOT" --alpha 10 --residual-ratio 0.5 \
  --inner-max-steps 5 --internal-nf-budget 1500 --replicate-tag prefix

"$CHIG_IR_PY" src/run_chignolin_ir_ivm_cpu1_repro.py \
  --root "$CHIG_IR_ROOT" --alpha 10 --residual-ratio 0.5 \
  --inner-max-steps 5 --internal-nf-budget 5000 --replicate-tag prefix
```

适配器内部固定 `OPENMM_CPU_THREADS=1`，其余算法参数和运行入口与主实验一致。

### 9.3 统一审计与严格前缀比较

```bash
"$CHIG_IR_PY" src/audit_chignolin_ir_ivm_results.py \
  runs/smoke/*.npz runs/screen/*.npz runs/production/*.npz runs/repro_cpu1/*.npz \
  --output-csv reports/final_all_optimization_audit.csv \
  --output-json reports/final_all_optimization_audit.json

"$CHIG_IR_PY" src/compare_cpu1_prefix.py \
  --short runs/repro_cpu1/ivm_chig_irivm_a1e1_r05_m05_inf1500_cpu1_prefix_central_state_ivm_cfg29d8ce2d7afa_budget1500.npz \
  --long runs/repro_cpu1/ivm_chig_irivm_a1e1_r05_m05_inf5000_cpu1_prefix_central_state_ivm_cfg844e61a3b6af_budget5000.npz \
  --output reports/cpu1_prefix_repro/prefix_comparison.json
```

参数汇总可由四个阶段的审计 CSV 重建：

```bash
"$CHIG_IR_PY" src/analyze_chignolin_ir_ivm_parameter_scan.py \
  reports/alpha_screen_r05_m05_inf1500/audit_summary.csv \
  reports/r_screen_a10_m05_inf1500/audit_all_screen.csv \
  reports/mmax_screen_a10_r05_inf1500/audit_summary.csv \
  reports/interaction_screen_a10_inf1000/audit_summary.csv \
  --output-dir reports/parameter_scan_combined
```

### 9.4 统一弧长 500 点势垒与 IVM 替换图

登记文件 `registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json` 固定五种方法的原始结果、哈希、时间状态选择和空间加密定义。完整物理重算可在干净输出路径中执行：

```bash
"$CHIG_IR_PY" src/postprocess_arc_length_500.py \
  --contract registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json \
  --output runs/diagnostics/chignolin_ivm_arc_length_500.npz \
  --workers 8
```

本次正式数据还保存了独立重算的 1182 条原始结果。若只需要严格复建标准化 NPZ 和逐行审计而不重复 591000 次诊断调用，可执行：

```bash
"$CHIG_IR_PY" src/postprocess_arc_length_500.py \
  --contract registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json \
  --output runs/diagnostics/chignolin_ivm_arc_length_500.npz \
  --precomputed runs/diagnostics/chignolin_arc500_independent_raw.npz \
  --overwrite
```

作图前先运行只读门禁，再生成请求的三联候选图：

```bash
ARC500=runs/diagnostics/chignolin_ivm_arc_length_500.npz
FIGDIR=/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/chignolin/active_candidates/chignolin_ir_ivm_cpu1

"$CHIG_IR_PY" src/plot_chignolin_ivm_three_metrics.py \
  --contract registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json \
  --arc500 "$ARC500" --output-dir "$FIGDIR" \
  --stem chignolin_ir_ivm_overlay_total_nf --dry-run

"$CHIG_IR_PY" src/plot_chignolin_ivm_three_metrics.py \
  --contract registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json \
  --arc500 "$ARC500" --output-dir "$FIGDIR" \
  --stem chignolin_ir_ivm_overlay_total_nf --replace
```

该脚本只重建候选目录中的 PDF/PNG/SVG、frontier CSV、annotations CSV 和 provenance JSON；论文副本只在候选 PDF 通过哈希与逐行门禁后以逐字节复制方式晋升，脚本不会直接修改 `main.tex`。

### 9.5 单元测试

```bash
"$CHIG_IR_PY" -m unittest discover -v tests
```

## 10. 结果解释边界

1. 905 是新图三个面板所用阈值下 IVM 的首次达到 NoF；在同一条已提交重参数化路径上，\(I_\perp=0.1852701724\)，\(\Delta E_{500}=24.0975823864\ \text{kcal mol}^{-1}\)，\(\widehat S=99.3776176017\ \text{kcal mol}^{-1}\)。
2. 37 组实验完整覆盖了 8 个 `alpha` 锚点以及选中 `alpha=10` 下的全部 30 个 `r × m_max` 组合；它不声称覆盖全部 240 个 `alpha × r × m_max` 组合。
3. 两线程结果用于参数趋势和候选筛选；候选图中的 IVM 使用可严格前缀复现、且与文章基线线程设置一致的单线程结果。旧 V9 IVM 已从图中删除。
4. recovery、被拒绝 trial 和终止尾段均计入 NoF；1182 条路径的 591000 次统一弧长 500 点诊断调用被明确排除在优化 NoF 之外。
5. 强制 `state_curve` 重参数化及 recovery 是 IR-IVM 身份的一部分。任何关闭、跳过或把未提交物理步作为结果的运行都不能用本文件的结论命名为 IR-IVM。
6. ODE-NEB 的 906 个保存状态和 OBSM-FIRE 的 168 个保存状态全部重算，因此两者的 \(\Delta E_{500}\) 首次达到是完整历史结果；Static-NEB 和 Original String 仅用文章登记的 48 个状态画势垒前沿，不对它们声称全历史的势垒首次达到，也不以它们画倍率箭头。
7. 2026-09-06 已将候选 PDF 的逐字节相同副本和机器可读侧车文件晋升到 `paper0819/Figures/chignolin/`，并更新 `paper0819/main.tex` 的 Chignolin 图、算法、指标与结果文字；验证产物为 `paper0819/tmp/vscode-latex/main.pdf`。
