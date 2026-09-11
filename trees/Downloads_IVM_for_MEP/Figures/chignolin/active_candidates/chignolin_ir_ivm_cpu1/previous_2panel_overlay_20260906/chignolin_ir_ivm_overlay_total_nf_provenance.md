# Chignolin IVM cost-to-precision 1x2 图溯源

首次登记：2026-09-01（Asia/Hong_Kong）

2026-09-04 新增独立的离散 geometric-action 诊断和附图，见 [geometric-action 溯源](chignolin_ivm_geometric_action_provenance.md)。原图的 PDF、500 点势垒数据与本登记的输入关系均保持不变。

## 1. 登记对象与状态

本文件只登记下面这张候选图：

`Figures/chignolin/active_candidates/chignolin_ir_ivm_cpu1/chignolin_ir_ivm_overlay_total_nf.pdf`

论文工作树根目录为：

`/Users/wuzhiyou/Downloads/IVM_for_MEP`

IR-IVM 实验根目录为：

`/Users/wuzhiyou/IVM_for_MEP/python/chignolin_ir_ivm_20260826`

状态：**active candidate，尚未进入 `main.tex`。** 登记时 `main.tex:1674` 仍引用
`Figures/chignolin/chignolin_cost_to_precision_total_nf_v9_20260730.pdf`。本次登记不修改正式图和论文正文。

图例中的 `IVM` 是本次选出的 IR-IVM 正式单线程结果；旧 V9 IVM 已从这张图中完全移除。

特别排除下面这个实验根目录中的同名历史 PDF：

`/Users/wuzhiyou/IVM_for_MEP/python/chignolin_ir_ivm_20260826/figures/active_candidates/chignolin_ir_ivm_cpu1/chignolin_ir_ivm_overlay_total_nf.pdf`

它的 SHA-256 是 `0be0fb35fda72e4fa49582f3f86b4a1c23afe821716f642d76f287e600adb473`，
属于旧 6 方法/旧 adaptive barrier 版本，不是本登记图，也不是当前图的数据来源。判断当前权威图时必须同时匹配
第 2 节的论文工作树路径和 SHA-256，不能只按文件名判断。

若人工文字与机器附件冲突，按以下优先级判断：

1. same-stem `chignolin_ir_ivm_overlay_total_nf_provenance.json`；
2. `registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json`；
3. `runs/diagnostics/chignolin_ivm_arc_length_500.npz` 及其 `.audit.json`；
4. 各方法的原始优化 NPZ、旁路 audit 和预登记；
5. same-stem `frontier.csv`、`annotations.csv`；
6. 本 Markdown 的解释文字。

## 2. 图及机器附件

| 角色 | 文件 | SHA-256 |
|---|---|---|
| 登记 PDF | `chignolin_ir_ivm_overlay_total_nf.pdf` | `1c023998e4359de0445ca3c7b108cd6b40753a9ac53a65fa4d6e154a55d87ed5` |
| 高分辨率预览 | `chignolin_ir_ivm_overlay_total_nf.png` | `38c24f2144a85f57467d99a99a5f3cf60e3b9f4375f38c2e1fab099ca82d74e6` |
| 矢量副本 | `chignolin_ir_ivm_overlay_total_nf.svg` | `d7a825775e09d76f1ca1c068078bda4e29c14db138de17d6dd0bf95f712d8b42` |
| 实际绘制的 Pareto 顶点 | `chignolin_ir_ivm_overlay_total_nf_frontier.csv` | `9d1d871620f8b6cf82ac4f6bfb8fc139e66dc4a7c7f9a5389341ae0669ed085f` |
| 箭头阈值与首次达到 | `chignolin_ir_ivm_overlay_total_nf_annotations.csv` | `8c74771d0dd9955b44a5c9ab18dbe8a855111c1dafc15a42d49b1c57593c1561` |
| 机器溯源 | `chignolin_ir_ivm_overlay_total_nf_provenance.json` | `cf83d9128ce68bcda1c39b3f2a0010466bde5589207ebddee0e75bf5de86c7ab` |

以上六个文件都位于本 Markdown 所在目录。PDF 为一页、约 180 mm x 72 mm，创建器为
Matplotlib 3.7.2。`frontier.csv` 保存图上每个实际曲线顶点的 method、metric、metric value 和
Number of Forces；`annotations.csv` 保存两根箭头的共同阈值、IVM/ODE-NEB 首次达到成本及未取整倍率；
`provenance.json` 保存输入路径、输入哈希、500 点协议、方法顺序和首次达到结果。

## 3. 最短完整数据链

```text
registered PDF / PNG / SVG
├── same-stem frontier.csv
│   └── 图中每个 strict Pareto 顶点：method + metric + value + Number of Forces
├── same-stem annotations.csv
│   └── 两个共同阈值、IVM/ODE-NEB 首次达到 NoF 和倍率
├── same-stem provenance.json
│   └── 五个原始结果路径/SHA、arc500 文件/SHA、协议和图输出哈希
└── src/plot_chignolin_ivm_arc500.py
    ├── 左图：五个原始优化 NPZ 的 i_perp + counter_vectors
    └── 右图：chignolin_ivm_arc_length_500.npz
        ├── 每行 source_result + source_record_index + path SHA + NoF
        ├── barrier_500_kcal
        ├── ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json
        └── src/postprocess_arc_length_500.py
            └── 原始优化路径 -> 全路径弧长均匀 500 点 -> OpenMM CPU/double
```

从 `frontier.csv` 的任意一行回溯时：先用 method 找到 `provenance.json` 中的
`source_results`；左图用 `(number_of_forces, i_perp)` 在原始结果的
`counter_vectors.sum(axis=1)` 和 `i_perp` 中定位，右图直接用 arc500 NPZ 中的
`method + source_record_index` 定位。arc500 NPZ 同时保存原始结果 SHA、路径 SHA、22 维计数向量和
`paper_total_nf`，因此右图每个指标点都能回到具体原始路径记录。

## 4. 两个面板怎样取数据

### 4.1 左图：\(I_\perp\)

左图不使用事后加密值。绘图脚本从每个原始结果读取 `i_perp` 和 22 图像的
`counter_vectors`，按

\[
N_F=\sum_{i=1}^{22}\texttt{counter\_vectors}[i]
\]

得到纵坐标，再按 Number of Forces 顺序保留严格改善的 \(I_\perp\) Pareto 前沿。横轴为逆向对数轴。

严格前沿的具体规则是：按 NoF 做稳定升序排列，依次维护当前最小指标值；只有指标严格低于此前最小值的记录才进入
`frontier.csv`。右图对 `barrier_500_kcal` 使用完全相同的 strict running-minimum 规则。

IVM 只允许初始路径和 `reparam_committed=True` 的完整路径进入曲线；其余方法的左图使用原始结果中的
全部保存历史。未提交的 direct/recovery trial 不作为科学路径状态，但其中发生的真实力调用仍已计入累计 NoF。

### 4.2 右图：\(\Delta E_{500}\)

右图只读取 `runs/diagnostics/chignolin_ivm_arc_length_500.npz` 的
`barrier_500_kcal`。每条选中路径被视为未加权 \(3N\) 笛卡尔空间中的分段直线，按累计弧长 \(s\) 取

\[
s_j=jL/499,\qquad j=0,\ldots,499,
\]

共 500 个总点，包含两个端点。每个点都直接调用一次新的 OpenMM 联合能量/力计算，不使用诊断缓存；势垒定义为

\[
\Delta E_{500}=\max_j E(x(s_j))-E(x(s_0)).
\]

方法的空间和时间取样如下：

| 方法 | 左图原始历史 | 右图 arc500 路径 | 说明 |
|---|---:|---:|---|
| IVM | 12 个可报告完整状态 | 12 | 初始状态或已提交重参数化状态 |
| ODE-NEB | 906 个保存状态 | 906 | 完整保存历史，右图首次达到对该历史精确 |
| Static NEB | 1000 个保存状态 | 48 | 右图沿用文章登记的 48 个 checkpoint，再统一重算 500 点 |
| Original String | 1000 个保存状态 | 48 | 右图沿用文章登记的 48 个 checkpoint，再统一重算 500 点 |
| OBSM-FIRE | 168 个保存状态 | 168 | 完整保存历史，右图首次达到对该历史精确 |

总计 1182 条路径，每条恰好 500 次诊断调用，共 591000 次。这些调用全部排除在图中的优化 NoF 之外。

## 5. 共同体系、初值和能量模型

| 项目 | 文件或定义 | SHA-256 |
|---|---|---|
| topology | `inputs/chignolin_clean.pdb` | `04b122a0b98492b6fadae35772ffc307b8916ea65c8eaf959cb5187bbbfeab01` |
| 初始路径容器 | `inputs/common_initial_path_equidistant.npz` | `18849a069d57f07ec1beb26eb79ca29c6b23d8881795e0d6a4ca97ebf9c6eba2` |
| 初始路径数组 | 所有五种方法的 `initial_path_sha256` | `57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996` |

上述相对路径以 IR-IVM 实验根目录为基准。路径共有 22 个图像，其中 20 个为可动内部图像；
每个图像有 166 个原子，展平路径形状为 `(22, 498)`。能量模型使用 OpenMM、
`amber14-all.xml`、`implicit/gbn2.xml`、`NoCutoff`、CPU 平台和 double precision。

## 6. 图中 IVM 的正式实验身份

正式实验登记：

`registrations/repro_cpu1/chig_irivm_a1e1_r05_m05_inf5000_cpu1_prefix.json`

SHA-256：`1cbe875c828b1a0591220fc9902ed60d332b2a631f4a476124187e766e87e16f`

正式原始结果：

`runs/repro_cpu1/ivm_chig_irivm_a1e1_r05_m05_inf5000_cpu1_prefix_central_state_ivm_cfg844e61a3b6af_budget5000.npz`

SHA-256：`e8ce449e55b975360bbc29fd49a8e28c1fe8c0618382eb37b0a86e022bdfe03a`

旁路 NoF 审计为同 stem `.audit.json`，SHA-256
`f6ae6899ef8cdf3ad8c09adcd6aad2891cec91a6ddc7a52e6d8f03414a314085`。

完整算法控制量由正式实验登记中的 expanded `command` 冻结。核心身份如下：

| 控制量 | 固定值 |
|---|---|
| 图中显示名 | `IVM`；源算法名为 IR-IVM |
| `alpha` | 10 |
| 离散相对残差阈值 `r` | 0.5 |
| 离散内部整带步数上限 `m_max` | 5 |
| 选中方向乘子 `eta` | 1.1 |
| L-BFGS history | 7 |
| frozen subproblem | `discrete_b040` |
| selector | `one_way_takeover` |
| 外层 globalization | `iperp_monotone` |
| 重参数化 | 每次外层迭代强制 `state_curve`，逐图像 direct acceptance，并保留 recovery |
| 物理最大原子位移 | 1 Angstrom |
| 内部 NoF 总预算 | 5000 |
| OpenMM 线程 | `OPENMM_CPU_THREADS=1` |
| 可报告路径 | 初始状态或 `reparam_committed=True` |

正式结果的 12 个可报告状态总 NoF 为：

`22, 126, 210, 386, 567, 735, 905, 1220, 1455, 1697, 1947, 3740`。

真实 cache miss 才计一次 NoF。所有被拒内部步、direct proposal、recovery 中间 trial 和回退都进入
22 图像累计账本；两个固定端点各包含一次初始调用。5000 内部预算对应最终总账 5002，最后一个未闭合事务已收费但不产生新的可报告路径。arc500 的 591000 次事后诊断不进入该账本。

## 7. IVM 参数选择和可复现性证据

正式 `(alpha, r, m_max)=(10,0.5,5)` 不是在画图时临时挑选。证据链为：

| 文件 | SHA-256 | 作用 |
|---|---|---|
| `registrations/CAMPAIGN_PREREGISTRATION.json` | `46c2649d693deae21305971434ac6c010d048405a5ee167965064536cbca7647` | OpenMM 运行前冻结算法、参数范围、计费和选择规则 |
| `registrations/CPU1_PREFIX_REPRO_PREREGISTRATION.json` | `2e6207db69b802a6651dab943a9ef21170d6019360f42bbb5218dc5ccdc83444` | 单线程正式复现前冻结唯一执行改动 |
| `reports/parameter_scan_combined/parameter_settings.csv` | `1e65a0ebb30bc539c86efd028723255fb4c8bb2b4a4d728a1316676ae49e35fe` | 37 组设置、结果路径/SHA、首次达到和右删失 |
| `reports/parameter_scan_combined/parameter_scan_analysis.json` | `7ebb284be8126084666f868db4b87789e57ff3a1eb5a24a79d85d46ed6d31684` | 参数规律和候选选择的机器分析 |
| `reports/final_all_optimization_audit.json` | `75d60efadbac4feb325516d9baaea5ec7e53492880ad0010840d9d10cfd9b7f8` | 43 个优化结果的完整 NoF/recovery 审计 |
| `reports/cpu1_prefix_repro/prefix_comparison.json` | `00038326686746c7fae3cbca2c97cbf2858d52b0f1c4d23ceae0f88f0a5e8b1a` | 单线程 1500/5000 结果的 70 个科学数组严格前缀一致性 |
| `reports/cpu1_prefix_repro/audit_summary.json` | `f6abfae1e591461d860162942650d28a7f8b993a3a6173e610062a117fdaf354` | 两个单线程结果的统一审计 |

37 组实验覆盖 8 个十倍尺度 `alpha` 锚点，并在 `alpha=10` 上完整覆盖 30 个
`r x m_max` 组合；它不是全部 240 组合的机械全展开。完整选择解释见本目录的
`RESULTS_AND_PROVENANCE.md`。

## 8. IVM 实验代码

以下源码哈希与正式单线程登记中的 `source_sha256` 一致：

| 代码角色 | 实验根目录相对路径 | SHA-256 |
|---|---|---|
| Chignolin/OpenMM backend、指标与共同函数 | `src/chignolin_core.py` | `ad5d6532f7acd7c7b2560fdb68e6ead796374bcf0b24d1c0790b7e51fa5e4faa` |
| 外层 IVM 与重参数化/recovery 状态机 | `src/run_ivm_v9_spring_projected_production_20260730.py` | `5a9a33625867b19200f20fca9675c3b3a3e1a25ca428b1e149397c7530474bd5` |
| 离散 frozen 子问题 | `src/frozen_discrete_b040.py` | `a42da93b10ab9e235770569159cfaf5bd36e819ed169ccb619729b5de6df9ed4` |
| FFrbk/L-BFGS one-way selector | `src/run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py` | `75f4dc5f4d7ab6c7af13e5b8e15a0f81b41b4a937346233a20a3e20e0de160f4` |
| `eta=1.1` 选中步乘子 | `src/run_ivm_v9_discrete_selected_eta_shim.py` | `a2f071ea787a008ed001fd806b8952f84d8db90335b544308478f3f6ae2facf9` |
| 三参数注入与配置身份 | `src/run_ivm_v9_discrete_ffrbk_takeover_grid_shim.py` | `4ee5cb4ad5536c7785dc6d017402af230f2451e1644cf7ef53d9de098757a6ae` |
| 总内部 NoF 硬预算 | `src/run_ivm_v9_total_internal_nf_budget.py` | `ca0fa9d44334094f406be6e3308b53326d9b81c18cc839feed831a184e764e58` |
| Chignolin 单设置运行器 | `src/run_chignolin_ir_ivm_member.py` | `d68f2db9857eb9b0f9d3bcfd2ba99261e34a216f81a0ee053d3d3e68030fee05` |
| 单线程正式复现适配器 | `src/run_chignolin_ir_ivm_cpu1_repro.py` | `b89bc82f13074ed3a81db2053ce7641291ae5d1e16ab49fab2ae57a7d9a3e779` |
| 结果与 NoF/recovery 审计 | `src/audit_chignolin_ir_ivm_results.py` | `11e91843e464f156b00f9c6a21374dd5f30a3db4080537c03d41cbd761d5ef88` |

## 9. 其余四种方法的数据来源

以下路径以论文工作树为基准。所有 raw NPZ 的文件 SHA 都由 arc500 比较合同再次固定；所有方法共享同一
`initial_path_sha256`。

| 方法 | 原始结果与 SHA-256 | 旁路 NoF audit 与 SHA-256 | 生产入口 |
|---|---|---|---|
| ODE-NEB | `code_on_server/chignolin/runs/neb/neb_paper_production_confirmed1000_ode_k50_central_k50_alpha0p0001_ode_budget1000.npz`<br>`68bc5c5a8608e173e650f194bf849127e583439862294f7afdae4fb4d03609ba` | 同 stem `.audit.json`<br>`c660c6530ac9abddfc409e89b04bf4c1bc7864a19ab8ed932e744ec848622839` | `code_on_server/chignolin/src/run_neb.py`，`k=50`，ODE optimizer |
| Static NEB | `code_on_server/chignolin/runs/neb/neb_paper_production_confirmed1000_static_k40_a0p012575_central_k40_alpha0p012575_static_budget1000.npz`<br>`8aa75a1cdee0cf82b2e46181c8633c6655277a18d37e9c738002546de0359fe0` | 同 stem `.audit.json`<br>`58001f77e9730317e58f3769303182afd7ddd976e993a45de8cef1834c62e54a` | `code_on_server/chignolin/src/run_neb.py`，`k=40, alpha=0.012575`，static optimizer |
| Original String | `code_on_server/chignolin/runs/confirmed1000_productions_20260728c/confirmed1000_productions_20260728c_paper_production_confirmed1000_original_dt0p012575_original_nolambda_dt0p012575_r1_cfg17accb085b5c_budget1000.npz`<br>`10778b9d36ffb6da7c2904cf024c5046137b6e103cd3581140ee440bd5ea5ea9` | 同 stem `.audit.json`<br>`abd95428d5ad648fd1b2f571112a21c245e26fdb0b4f15fc504a21bfde35770e` | `code_on_server/chignolin/src/run_string_family.py`，`dt=0.012575`，每步重参数化 |
| OBSM-FIRE | `code_on_server/chignolin/runs/confirmed1000_productions_20260728c/confirmed1000_productions_20260728c_paper_production_confirmed1000_obsm_l0p9_n5_dt0p12_fire_lam0p9_inner5_r1_nocap_cfgb34d772e4eb2_budget1000.npz`<br>`0dc20cbb94fa5411d8bd6bd8d40f303766937c5327a7fbc2f965b71668bf7b4c` | 同 stem `.audit.json`<br>`9feacd1cea18c553e51f459f6d62da14d83849bd53bfdd1d39496767a7daebe4` | `code_on_server/chignolin/src/run_obsm.py`，`lambda=0.9, n_inner=5, dt=0.12`，FIRE |

基线的精确命令、参数、结果路径、audit 路径和记录数由
`code_on_server/chignolin/figures/article_update_20260730_v9_final/chignolin_method_comparison_summary.json`
冻结；该文件 SHA-256 为
`7d6c593ac70fe311d345364435e38d2941d494172c2dc70acfbfac922cb33754`。

当前保存的基线入口源码快照哈希为：

| 源码 | SHA-256 |
|---|---|
| `code_on_server/chignolin/src/run_neb.py` | `de73a97cef2e1ff9771e96bcc1b65fa080ba12dc33712be8761f38a7415eda28` |
| `code_on_server/chignolin/src/run_string_family.py` | `8ead7ba050ce6dfbe2535091b302d61f2705d5cbdb9cab717425da7e5e7c11ad` |
| `code_on_server/chignolin/src/run_obsm.py` | `9ba31d8bab402b80c59b556ed84c0db5acc07b5720e25425cdb0bd8473a38834` |
| `code_on_server/chignolin/src/chignolin_core.py` | `ad5d6532f7acd7c7b2560fdb68e6ead796374bcf0b24d1c0790b7e51fa5e4faa` |

Static NEB 与 Original String 的 48 个右图 checkpoint 分别由旧登记文件
`dense_barrier_paper_total_nf_confirmed1000_static_k40_a0p012575_adaptive48.npz`
和 `dense_barrier_paper_total_nf_confirmed1000_original_dt0p012575_adaptive48.npz` 提供调用编号；它们的 SHA-256 分别为
`5fda5bce0e2e4dc6f2397d7075b0f238d41010079b722daae3e24582f61cb0a3` 和
`b0e1b0edfd3bef041b545e674e4f81f286f834320e8881f097e7d2d37feafc9c`。
旧文件只决定选哪些历史时点；本图不读取其中的旧势垒值，所有右图势垒都重新用统一 arc500 协议计算。

## 10. 500 点指标代码与数据

| 角色 | 实验根目录相对路径 | SHA-256 |
|---|---|---|
| 500 点比较合同 | `registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json` | `f351020439a1ef7d75bd0c21ae6d843722d813912542bf5cef984f3f0ab424b8` |
| 独立原始重算 | `runs/diagnostics/chignolin_arc500_independent_raw.npz` | `15e4ed61e0e9c213055060566b98ee4739cc47263c2b5ab53f881153fe3a81f1` |
| 独立原始重算审计 | `runs/diagnostics/chignolin_arc500_independent_raw.json` | `c0adb3aa0f37692abd719f90be7afa8de6e1c88a007361885e2f4e33acddc1b7` |
| 作图读取的标准化指标 | `runs/diagnostics/chignolin_ivm_arc_length_500.npz` | `3def8ffb3c70342d61285813511203c2970e30cb0146f37aa1d99aa3ce20a564` |
| 标准化指标逐记录审计 | `runs/diagnostics/chignolin_ivm_arc_length_500.audit.json` | `03bca53e4390fd2a0373901ef7cea3897ffb4ff69717418924306defd17cd509` |
| 500 点重算/导入门禁 | `src/postprocess_arc_length_500.py` | `607c0ceb41f3de4f6cf40d8094f37947bb687f72d4c516be219453cb5bbe4a17` |
| 500 点和首次达到单元测试 | `tests/test_arc_length_500.py` | `dc387d3b6e9638b76182493119f43e83afaedc2db075b2141fd17569722a4cd2` |

标准化脚本可以直接执行 591000 次 OpenMM 诊断，也可以导入独立原始重算；导入时仍逐行验证
method、record index、NoF、\(I_\perp\)、path SHA、arc length 和 500-call 声明，缺一项即拒绝输出。

`chignolin_arc500_independent_raw.*` 的角色是独立交叉审计。比较合同保存了当时两个一次性独立脚本的 SHA-256，
但没有保存可访问的脚本路径；因此不能把这两个 raw 文件单独宣称为完整代码可复现入口。当前权威、已归档且可从五个原始
NPZ 重新执行全部物理计算的入口是 `src/postprocess_arc_length_500.py`。这项边界在本登记中显式保留，避免把“独立数值核对”
误写成“独立代码已完整归档”。

## 11. 绘图代码、样式与数值结果

唯一当前绘图入口为：

`src/plot_chignolin_ivm_arc500.py`

SHA-256：`62d30b5b81e0ab3b42bdf080308e95c615a4ecee0b960946b1df4dd15e0743b9`

它在出图前重新验证比较合同、五个原始结果 SHA、共同初始路径、arc500 NPZ/audit 及 NoF 对齐。
图形样式参照
`/Users/wuzhiyou/IVM_for_MEP/output/pdf/alanine66d_delta_aware_grid_ivm_replacement_1x2_preview.pdf`，
该参考 PDF SHA-256 为
`483530e8c0d764e2c4b4c82af3c280cc7f8fee300eb35b12db6e1260759f3a44`。

共同阈值定义为 IVM、ODE-NEB、OBSM-FIRE 三种方法都达到的最严格精度。图中结果为：

| 指标 | 共同阈值 | IVM 首次达到 NoF | ODE-NEB 首次达到 NoF | OBSM-FIRE 首次达到 NoF | 图中 ODE-NEB/IVM |
|---|---:|---:|---:|---:|---:|
| \(I_\perp\) | 0.21242215133173672 | 905 | 17662 | 19942 | 19.5x |
| \(\Delta E_{500}\) (kcal/mol) | 24.17608142601358 | 905 | 15842 | 19942 | 17.5x |

IVM 在 NoF 905 的实际值为
\(I_\perp=0.1852701724454184\) 和
\(\Delta E_{500}=24.097582386424108\ \mathrm{kcal/mol}\)。两幅图只画 IVM 与 ODE-NEB 的倍率箭头；
OBSM-FIRE 参与共同阈值定义并保留曲线，但不画倍率箭头。

具体地，对指标 \(q\) 先求三种参考方法各自保存历史中的最小值，再取
\(q_\mathrm{shared}=\max_m \min q_m\)，保证三种方法都实际达到该精度；首次达到成本是按 NoF 排序后的
running minimum 第一次不大于 \(q_\mathrm{shared}\) 的 NoF。图中倍率严格等于
`ODE-NEB first-passage NoF / IVM first-passage NoF`，最后才显示为一位小数。

## 12. 复现顺序

现有正式结果和登记带防覆盖门禁。不要在原路径上静默重跑；完整重跑应使用新的输出目录或新的
replicate tag。以下命令给出相同代码路径下的复现顺序：

```bash
CHIG_IR_ROOT=/Users/wuzhiyou/IVM_for_MEP/python/chignolin_ir_ivm_20260826
CHIG_IR_PY=/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python
FIGDIR_REPRO=/private/tmp/chignolin_ivm_overlay_repro
cd "$CHIG_IR_ROOT"
```

正式 IVM 实验的简化入口；完整 expanded command 以单线程正式登记 JSON 为准：

```bash
"$CHIG_IR_PY" src/run_chignolin_ir_ivm_cpu1_repro.py \
  --root "$CHIG_IR_ROOT" --alpha 10 --residual-ratio 0.5 \
  --inner-max-steps 5 --internal-nf-budget 5000 --replicate-tag NEW_TAG
```

在新的诊断输出路径上完整重算统一 arc500：

```bash
"$CHIG_IR_PY" src/postprocess_arc_length_500.py \
  --contract registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json \
  --output runs/diagnostics/chignolin_ivm_arc_length_500_REPRO.npz \
  --workers 8
```

若只复建当前标准化指标而不重复 591000 次 OpenMM 调用，可在新的输出名下严格导入独立原始重算：

```bash
"$CHIG_IR_PY" src/postprocess_arc_length_500.py \
  --contract registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json \
  --output runs/diagnostics/chignolin_ivm_arc_length_500_REPRO.npz \
  --precomputed runs/diagnostics/chignolin_arc500_independent_raw.npz
```

先对登记数据执行只读作图门禁，再在新的临时目录和新 stem 下生成复验图；下面的安全命令不会覆盖当前候选图：

```bash
"$CHIG_IR_PY" src/plot_chignolin_ivm_arc500.py \
  --contract registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json \
  --arc500 runs/diagnostics/chignolin_ivm_arc_length_500.npz \
  --output-dir "$FIGDIR_REPRO" --stem chignolin_ir_ivm_overlay_total_nf_repro --dry-run

"$CHIG_IR_PY" src/plot_chignolin_ivm_arc500.py \
  --contract registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json \
  --arc500 runs/diagnostics/chignolin_ivm_arc_length_500.npz \
  --output-dir "$FIGDIR_REPRO" --stem chignolin_ir_ivm_overlay_total_nf_repro
```

测试命令：

```bash
"$CHIG_IR_PY" -m unittest discover -v tests
```

## 13. 解释边界

1. 本文件登记的是候选图，不代表已修改论文正式结果。
2. 图中 `IVM` 的数据来自 IR-IVM `(10,0.5,5)`；旧 V9 IVM 不在图内。
3. 右图的所有势垒都使用同一 500 总点弧长协议；旧 adaptive dense 数值不进入右图。
4. Static NEB 和 Original String 的右图只覆盖文章登记的 48 个 checkpoint，因此不对它们声称完整历史的势垒首次达到，也不画倍率箭头。
5. 591000 次 arc500 诊断是事后评价，不能加到或减去图中的优化 NoF。
6. 更完整的参数扫描、recovery 账本与科学解释见本目录的 `RESULTS_AND_PROVENANCE.md`；本文件负责把这一张图的输入和生成链集中登记。
