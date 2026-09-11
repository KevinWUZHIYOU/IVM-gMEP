# Barnase--Barstar IVM 实验与最终图溯源

> 状态：**active candidate**。本登记对应 Barnase--Barstar 上由 Chignolin IR-IVM 最小必要迁移得到的 selected IVM，以及与 official ODE-NEB 的统一指标比较。两张图已生成并通过机器审计；当前未修改论文 `main.tex`。

## 1. 登记对象与证据优先级

实验根目录：

`/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901`

论文工作树：

`/Users/wuzhiyou/Downloads/IVM_for_MEP`

| 图 | PDF | same-stem provenance | 状态 |
|---|---|---|---|
| IVM/ODE-NEB cost-to-precision，1x2 | `/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/barnase_barstar/active_candidates/barnase_barstar_ivm_vs_ode_neb_cost_to_precision.pdf` | 同目录 `barnase_barstar_ivm_vs_ode_neb_cost_to_precision.json` | 已生成，active candidate |
| Barnase--Barstar IVM evolution，1x4 | `/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/barnase_barstar/active_candidates/barnase_barstar_ir_ivm_evolution.pdf` | 同目录 `barnase_barstar_ir_ivm_evolution.json` | 已生成，active candidate |

| 文件 | SHA-256 |
|---|---|
| cost PDF | `616d0a5d34f739d12d310abe34ece044c526b969c27dadb3e5cb6594f3b09a7b` |
| cost provenance | `1e5dbec1a5fb54aceb2a2658c1083715e787b6f212565b2659dfa3d3f4799203` |
| cost figure contract | `c10c3be8af4352e8658e7f2345bacdfa5e16edb574f949f924de3e4500f4ab38` |
| evolution PDF | `9f83d9146b26b4ca90d17470db6bf4b4d8f5029d7b04eedd493efa66bbf81da0` |
| evolution provenance | `39887443b9d43640199d0ae89fe12191c32c24c60c5fc32b50ca63d6b0da351b` |
| evolution figure contract | `269b7195e8c7c1185abc5479bc1c244c657d26c96f42a1123f17b8ef784f3a5a` |

若本登记文字与机器附件冲突，按以下顺序判定：same-stem provenance、最终 figure contract、指标 audit/verifier、原始 NPZ/audit、本 Markdown。

## 2. 正式 IVM 实验身份

正式成员标签为 `bb_irivm_a1e0_r01_m05_inf9990_r1`。图中按文章语言统一显示为 `IVM`。

| 控制量 | 冻结值 |
|---|---:|
| 图像数 | 32；30 个可动内部图像，两个 fixed endpoints |
| \(\alpha\) | 1 |
| 离散相对残差阈值 \(r\) | 0.1 |
| 离散内部整带步数上限 \(m_{\max}\) | 5 |
| 选中步乘子 \(\eta\) | 1.1 |
| L-BFGS history | 7 |
| 最大原子位移 | \(1\,\mathrm{\AA}\) |
| 重参数化 | 每次外层迭代使用 `state_curve`；保留 per-image direct acceptance 与 recovery |
| 初始路径 | hash-bound `inputs/common_initial_path32.npz`；禁止 warm start |
| 注册优化预算 | \(N_F\le9990\)，是硬上界而不是必须达到的目标 |
| 实际终态优化 \(N_F\) | 9985 |
| fixed-endpoint 优化 \(N_F\) | 0 |
| 可报告路径 | 初始路径或 fully reparameterization-committed path |

正式登记、授权与启动收据：

| 角色 | 路径 | SHA-256 |
|---|---|---|
| member registration | `registrations/production/bb_irivm_a1e0_r01_m05_inf9990_r1.json` | `f6168c528554e9e4055732add7a825a9acdb5a5fecc02a58abb88c18a534af86` |
| execution authorization | `registrations/execution_authorizations/bb_irivm_a1e0_r01_m05_inf9990_r1.json` | `5306d0cd6470068a9719383b567761614022391b02ebdc3d47812d1121dd6785` |
| launch receipt | `runs/lock_receipts/bb_irivm_a1e0_r01_m05_inf9990_r1.launch.json` | `295f6977247f58a7e67badfd01ccbacc542055ddb287ffab19a5049b54c794ae` |
| selected design | `registrations/BB_IRIVM_SELECTED_9990_DESIGN_20260902_R1.json` | `e36b596113fc1dd8cfddf1aa37efb124182c5c7dcd2aedcdf88e7253ebc182e2` |

Launch receipt 的终态是 `CHILD_EXITED_LOCKS_RELEASED`，return code 为 0。

## 3. 最短完整数据链

```text
selected member registration + execution authorization + launch receipt
└── terminal optimization NPZ + producer audit
    ├── terminal NoF verifier: PASS
    ├── exact 4080-prefix verifier: PASS_EXACT_PREFIX
    └── selected arc500 contract
        └── selected arc500 NPZ + audit: PASS
            ├── 每条可报告路径的 discrete I_perp
            └── 同一条路径的 arc-length-500 Delta E

official ODE-NEB 333-paid-state ledger
├── milestone arc500: NF 240, 480, 990, 4080, 9990
├── targeted round 1: NF 8160, PASS
└── targeted round 2: NF 2040, PASS

selected/ODE metric bundles + immutable final figure contracts
├── cost plotter -> cost PDF + same-stem provenance
└── evolution plotter -> evolution PDF + same-stem provenance
```

Selected IVM 的权威终态闭包：

| 角色 | 路径 | SHA-256 | 状态 |
|---|---|---|---|
| terminal result | `runs/production/ivm_bb_irivm_a1e0_r01_m05_inf9990_r1_central_state_ivm_cfge0354bb7178e_budget9990_storagecommitted.npz` | `117f022589311ed75d27a5b6bfd36bca37b3da43b5b3f4f5fcb8bcdc40cb7df4` | terminal |
| producer audit | 同 stem `.audit.json` | `1d82aa3026faf2834ce1050de7c8454dab17c51433a1574563c24742772b6e39` | counters agree |
| terminal verifier | `reports/verifications/BB_IRIVM_A1E0_R01_M05_INF9990_R1_VERIFICATION_20260902.json` | `af362a3db925399a5bdc49a5107f8d1988341d0ee3fd0dbfed20d31ae628b584` | `PASS` |
| exact-prefix verifier | `reports/verifications/BB_IRIVM_SELECTED_9990_EXACT_4080_PREFIX_20260902_R1.json` | `182d096300578fc2cc192b5fc6245b69e15da24c877c0ef31f717087245526c1` | `PASS_EXACT_PREFIX`，无 tolerance fallback |
| selected arc500 contract | `registrations/diagnostics/BB_IRIVM_SELECTED_9990_ARC500_20260902_R1.json` | `ab7ee74ad264794434bd43e82ffb63fbe8e5ab3ca8523854ac1023fcf8ccf027` | frozen |
| selected arc500 result | `runs/diagnostics/bb_irivm_selected_9990_arc500_20260902_r1.npz` | `26244693b7752d8f61e7bbf706a3cea0f4a2ac9fa30b9e99fafce9fe2cfefb8f` | complete |
| selected arc500 audit | 同 stem `.audit.json` | `be5fbbc0f20777216a1d7facb80ba51b94f9870dcd8292d913a21ceb0ac54a7d` | `PASS` |

## 4. Number of Forces \(N_F\)

优化横轴 `Number of Forces` 统计 30 个内部图像发生的每一次物理 joint OpenMM `getState` 尝试，包括离散内部 trial、后来被拒绝的 trial、重参数化 direct proposal 和 recovery 的选择性中间 trial。两个 fixed endpoints 使用预先绑定的 E/F cache，因此 endpoint 优化 \(N_F=0\)。Arc500 事后能量评价与优化 \(N_F\) 分账，贡献严格为 0。

### 4.1 为什么 4056 不是 30 的整数倍

4080 上限参考实验的实付优化成本为

\[
\begin{aligned}
4056
&=\bigl[(1+128)\times30-7\bigr]
 +3\times30
 +(30+39+34)\\
&=3863+90+103.
\end{aligned}
\]

解释：

1. 初始整带评价加 128 个离散内部整带 trial 名义上是 \(129\times30=3870\)；7 次内部坐标精确命中 cache，不发生新物理调用，所以实付 3863。
2. 三次 direct reparameterization proposal 各调用 30 个内部图像，共 90。
3. 三个 recovery 序列分别发生 30、39、34 次选择性物理调用，共 103。

Recovery 会让不同 image 独立停止，也可能只继续评价部分 image，因此它的调用数不要求是 30 的整数倍。四套独立账本均为 4056；不存在漏报。

### 4.2 正式终态 9985 的完整分解

\[
\begin{aligned}
9985
&=\bigl[(1+325)\times30-145\bigr]
 +4\times30
 +(30+39+34+127)\\
&=9635+120+230.
\end{aligned}
\]

- 初始整带加 325 个离散内部整带 trial，扣除 145 次内部精确 cache hit，实付 9635；
- 四次 direct proposal 实付 120；
- 四个 recovery 序列实付 \(30+39+34+127=230\)。

| 权威账本 | 终态值 |
|---|---:|
| physical joint attempts | 9985 |
| successful joint evaluations | 9985 |
| 30 个内部图像 cache misses 之和 | 9985 |
| reported optimization internal \(N_F\) | 9985 |

最后一个可报告 committed path 在 \(N_F=8982\) 获得。之后 \(9985-8982=1003\) 次调用仍真实计费，但没有形成新的 fully committed path，不能伪造成 \(N_F=9985\) 的指标点。

## 5. 两个主指标的统一协议

### 5.1 离散 normal residual \(I_\perp\)

在同一个 32-image path 上计算。内部切向量使用归一化 central chord \(x_{i+1}-x_{i-1}\)，离散求积采用 half-adjacent-chord 权重，两个 fixed endpoints 的 force contribution 为 0。力来自该状态已经付费的 30 个内部图像，不新增物理调用。单位为 \(\mathrm{eV^2/\AA}\)。

### 5.2 Densified forward barrier \(\Delta E\)

对与 \(I_\perp\) 完全相同的 32-image path，在未加权 \(3N\) Cartesian 空间按累计弧长均匀取 500 个总点，包含两个端点：

\[
s_j=jL/499,\qquad j=0,\ldots,499.
\]

路径采用 piecewise-linear interpolation，并计算

\[
\Delta E=\max_j E(x(s_j))-E(x(s_0)).
\]

每条新路径必须有 500 次成功 energy-only 评价、0 次 force request、0 次 optimization \(N_F\) contribution，并使用相同的 OpenMM 8.1.2 CPU/double Amber14/GBn2、NoCutoff oracle。只有路径、500 点网格、oracle 和能量向量逐项 hash-identical，且来源 audit 为 `PASS`，才允许精确复用。Online sparse-image barrier 不参与选参、比较或最终作图。

## 6. Selected IVM 的统一指标

| 可报告路径 | \(N_F\) | discrete \(I_\perp\) | densified \(\Delta E\) (kcal mol\(^{-1}\)) |
|---|---:|---:|---:|
| common initial path | 30 | \(3.310616093\times10^9\) | 32632.032801 |
| committed path 1 | 210 | \(8.483398585\times10^6\) | 10814.684022 |
| committed path 2 | 429 | 77444.135642 | 900.100684 |
| committed path 3 | 1993 | 440.840868 | 613.463371 |
| committed path 4 | 8982 | 34.372335 | 686.288089 |

\(\Delta E\) 不是单调量：最后一个 path 的 \(I_\perp\) 最低，但 \(\Delta E\) 高于 \(N_F=1993\) path。不得把 \(N_F=8982\) 的 \(I_\perp\) 与 \(N_F=1993\) 的 \(\Delta E\) 拼成一条不存在的路径状态。

## 7. ODE-NEB 的统一指标与 targeted lineage

| \(N_F\) | discrete \(I_\perp\) | densified \(\Delta E\) (kcal mol\(^{-1}\)) | 来源 |
|---:|---:|---:|---|
| 240 | \(3.430974610\times10^7\) | 25282.607135 | milestone PASS |
| 480 | \(3.818398991\times10^6\) | 11652.176101 | milestone PASS |
| 990 | 129531.445712 | 2171.949991 | milestone PASS |
| 2040 | 21120.477930 | 1110.626908 | targeted round 2 PASS |
| 4080 | 4371.165337 | 912.372820 | milestone PASS |
| 8160 | 1387.247929 | 2866.723138 | targeted round 1 PASS |
| 9990 | 631.904399 | 1025.689622 | milestone PASS |

| 数据 | selection SHA-256 | result SHA-256 | audit SHA-256 |
|---|---|---|---|
| milestone NF 240/480/990/9990 | 不适用 | `b8feb051df121a29da01c93c6c0a2e1324dc189a07b5d44e6c8b3269b960e989` | `22115e01097ea77c66a0ab2801cdce2c04e4e9e2d13a7e6dfd58388a89e94ee1` |
| milestone NF4080 | 不适用 | `66562b7d4f4f55eb97d12a9d76f79957c52266a1c952226fb657a051a8ee2be2` | `10bc0bc4c9a08a5bbc71e0d5697364b781bbd6aa83ebe271b4078522d71cb82e` |
| targeted NF8160 | `b8dd0b1ccf597945ac5ae95c2ead177bbee778d657ba78bbb3c266648c46a3fd` | `1beff260e797349c1101b51a17dd72e272f7bc434c441df277a7b4fad602840b` | `f7ce2fed4a2546992e88f0c4d9e2c29d1118e8a1e249180ea3f01948b97dadd1` |
| targeted NF2040 | `8b5762aad446708b4f3cf3a68b588cf7c198a88acf22f6fa3ae99b922f445d24` | `ec6e598767e2917c0137b0fd60b672eb32100ef8457d98eef518f1eaf0af081c` | `f55f06fb5f848f77ad28df1126422e0d2dd3f41af15ec6850b070c728202b9e0` |

两轮 targeted audit 均确认每条路径 500 fresh energy attempts、500 successes、0 force requests、0 optimization \(N_F\) contribution，且离散 \(I_\perp\) 绑定到 already-paid force receipt。

### 7.1 NF2040 的零调用失败记录

第一次 NF2040 启动因远程命令遗漏六个 frozen thread caps，在构造 energy evaluator 时被门禁拦截。失败收据为：

`runs/diagnostics/bb_ode_neb_targeted_nf2040_arc500_20260902_r2_prephysical_failure.json`

SHA-256 为 `c68b4d0d9f69aa69b3530702b80863df61136b22f7704a5eb2db67101ff96ca1`。

失败 WAL 只有 `header` 和 `path_start`，没有 `attempt_started`；energy/force attempts 都是 0，没有 NPZ 或 PASS audit。旧 WAL 原样保留且不覆盖。随后用新合同 `BB_ODE_NEB_TARGETED_NF2040_ARC500_EXECUTION_20260902_R2A2.json`（SHA-256 `bf129e39690169da39ef932201cffd594d888f2d1296857b265337da2e90188a`）和新 output stem 完成正式计算。因此第一次启动既不计诊断能量，也不污染优化 \(N_F\)。

## 8. Cost-to-precision 的最终数值

曲线定义为：达到每个更低已记录指标值所需的最小观测优化 \(N_F\)。全部原始行先保留，再构造 lower envelope；非单调 \(\Delta E\) 行不会被从 provenance 删除。

| 指标 | 共同阈值 | IVM 首次观测达到 \(N_F\) | ODE-NEB 首次观测达到 \(N_F\) | ODE-NEB / IVM |
|---|---:|---:|---:|---:|
| discrete \(I_\perp\) | 631.904399 | 1993 | 9990 | 5.0125439×，图中 5.0× obs. |
| densified \(\Delta E\) | 912.372820 kcal mol\(^{-1}\) | 429 | 4080 | 9.5104895×，图中 9.5× obs. |

这里的倍率是 observed-passage cost ratio，不是 exact first-passage ratio。ODE-NEB 尚未对所有更早 distinct retained paths 进行 arc500：

- \(I_\perp\) 的可能 exact first-passage 区间为 \([30,9990]\)，观测首达之前仍有 277 条未测路径；
- \(\Delta E\) 的可能 exact first-passage 区间为 \([30,4080]\)，观测首达之前仍有 112 条未测路径。

因此图中明确使用 `obs.`，不假设 \(\Delta E\) 随 \(N_F\) 单调，也不做未测路径的插值或外推。

## 9. 与 current BB IVM 的独立双指标判定

Current BB IVM reference 为 `runs/diagnostics/bb_original_ivm_arc500_20260901_r1.npz`（SHA-256 `909bd969e1001ab8f6002cbb58791ffe8ca24e5f612359e6add7a72a807ad244`），其 audit SHA-256 为 `4a1376c1c6afb03a2905c79c9cc823fb433aff43644edcebbc9b535c52640d31`，状态为 `PASS`。

必须用同一条 selected path 上的两个指标比较：

| 方法/路径状态 | \(N_F\) | discrete \(I_\perp\) | densified \(\Delta E\) |
|---|---:|---:|---:|
| selected IVM，committed path 3 | 1993 | 440.840868 | 613.463371 |
| current BB IVM reference | 9990 | 1464.482266 | 628.327100 |

Selected IVM 在同一条 \(N_F=1993\) path 上同时改善两个主指标：\(I_\perp\) 改善 3.3220×，\(\Delta E\) 降低约 2.37%，而 current BB IVM 的成本是它的 \(9990/1993=5.0125\) 倍。不能用 \(N_F=8982\) 做“双指标都更好”的 comparator，因为该路径的 \(\Delta E=686.288089\) 高于 current BB IVM 的 628.327100。

## 10. 实验代码、指标代码与画图代码

| 角色 | 路径 | SHA-256 |
|---|---|---|
| BB/OpenMM backend | `src/bb_core.py` | `8cc9f3519bd36ef46901dd86ec2cdd9e8005763993892d48a9daf3d24e4b9f2d` |
| 外层 IVM、重参数化/recovery | `src/run_ivm_v9_spring_projected_production_20260730.py` | `991c11ff1f8eab014a6e54b4bf8ba610b0e52d071b202dc131b03bfabe6143fa` |
| discrete frozen subproblem | `src/frozen_discrete_b040.py` | `a42da93b10ab9e235770569159cfaf5bd36e819ed169ccb619729b5de6df9ed4` |
| FFrbk/L-BFGS selector | `src/run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py` | `75f4dc5f4d7ab6c7af13e5b8e15a0f81b41b4a937346233a20a3e20e0de160f4` |
| \(\eta=1.1\) shim | `src/run_ivm_v9_discrete_selected_eta_shim.py` | `a2f071ea787a008ed001fd806b8952f84d8db90335b544308478f3f6ae2facf9` |
| 参数注入 | `src/run_ivm_v9_discrete_ffrbk_takeover_grid_shim.py` | `b93a0156bd05cf1f1eafced5a07d608160e2f9ec440136c2ed5961a240709d2b` |
| 内部 \(N_F\) 硬预算 | `src/run_ivm_v9_total_internal_nf_budget.py` | `71e82c4d32b07e9295b44e2af18b9b6c0e7dbad0d5cbc4a517288ce213d03e22` |
| 单成员入口 | `src/run_barnase_barstar_ir_ivm_member.py` | `ebb9d2c52d5e3a12c477c0e735797d9aff5fee5df311b8c84f87f0b39f4cfd85` |
| arc500 指标 evaluator | `scripts/evaluate_discrete_iperp_densified_barrier.py` | `ab56602fa94e3ab5f03b38b5210fc99430e128491ff3ea1e0b5967b21a2f5304` |
| targeted ODE selector | `scripts/freeze_targeted_ode_neb_arc500.py` | `f7ac46f21057fee888eb4d3af8c9aeacfde0af4dbbf1d8545f2ce23e49ce1242` |
| targeted ODE evaluator | `scripts/evaluate_targeted_ode_neb_arc500.py` | `4dd9c3f713b9d0729945c65d562652d13318ab626bfb25931a89622f109d157c` |
| cost plotter | `scripts/plot_bb_cost_to_precision_final.py` | `50c1e7f7e9c99eec2dcb81c4d50bad6efcb022e6dcdfb5cc3a8ab90981b7117a` |
| evolution plotter | `scripts/plot_bb_irivm_evolution_hashbound.py` | `c894ce6f4e0cf6f6509c12dc2fc3d2297c61ea55c9523bfb908380174ea63952` |

迁移范围与禁止算法漂移由 `contracts/MIGRATION_CONTRACT.json`（SHA-256 `37126d31a286ef308cd4bc9cae581fc07b7ba876eb4cce552260ff961b493ccc`）冻结。统一指标由 `registrations/USER_METRIC_POLICY_AMENDMENT_20260901_R1.json`（SHA-256 `ca547dd86a119e0dd4b6049e296ae9616d335248c3df1ce1f77f7fcb91500bc5`）冻结。

## 11. 复现入口

Selected terminal verification、exact-prefix verification、arc500 contract 与 arc500 evaluation 的顺序及完整命令冻结在：

`registrations/diagnostics/BB_IRIVM_SELECTED_9990_POSTRUN_PIPELINE_20260902_R1.json`

SHA-256 为 `64845d568f8b88a69412f6bfaa350176ccedf297f48ad5b8509872e0e1b987d8`。

NF2040 targeted round 2 的正式命令从实验根目录执行：

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENMM_CPU_THREADS=1 \
PYTHONPATH=src:scripts \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
scripts/evaluate_targeted_ode_neb_arc500.py \
--selection registrations/diagnostics/BB_ODE_NEB_TARGETED_ARC500_SELECTION_20260902_R2.json \
--execution-contract registrations/diagnostics/BB_ODE_NEB_TARGETED_NF2040_ARC500_EXECUTION_20260902_R2A2.json \
--output runs/diagnostics/bb_ode_neb_targeted_nf2040_arc500_20260902_r2a2.npz
```

作图前已用同一组输入运行 `--validate-only`。Cost PDF 的完整实际 `command_argv`、全部输入 SHA、原始指标点、lower envelopes、共同阈值、claim scope 与未取整倍率都记录在 same-stem provenance JSON；不依赖 shell history 重构。

## 12. PDF 与登记 QA

- cost PDF：1 页，510.218 × 203.891 pt；所有四种字体均 embedded/subset；渲染检查无裁切、错位、重复 panel label 或不可读文字。
- evolution PDF：1 页，515.433 × 184.026 pt；字体均嵌入；渲染检查无注释裁切或 panel 错位。
- cost 图只比较 `IVM` 与 `ODE-NEB`；纵轴为 `Number of Forces`；倍率位于双向箭头中点右侧。
- 两张图的 same-stem provenance 中 PDF SHA 与文件实算一致。
- 最终 cost figure contract 状态为 `FROZEN_FINAL_INPUT_BINDINGS`，无占位符，并绑定两轮 targeted selection/result/audit。
- 本登记无尖括号占位符；`main.tex` 未修改，因此两张图保持 `active candidate`。

