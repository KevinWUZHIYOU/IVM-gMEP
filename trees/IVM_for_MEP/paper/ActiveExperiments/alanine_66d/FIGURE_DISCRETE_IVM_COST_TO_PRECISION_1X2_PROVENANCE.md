# Alanine 66D one-way alpha15 eta1.1 single-run discrete IVM 1x2 图溯源

首次登记：2026-08-23；短文件名替换与标签更新：2026-08-24（Asia/Shanghai）

## 1. 登记对象、状态和优先级

本文件只登记下面这一张图：

`paper/ActiveExperiments/alanine_66d/figures/active_candidates/alanine66d_discrete_ivm_cost_to_precision_1x2.pdf`

这个短 stem 已取代原来的
`alanine66d_discrete_single_one_way_a15_eta1p1_cost_to_precision_1x2_all_methods_muller_style`
三件套；旧 PDF、`_source.csv` 和 `_manifest.json` 已从 active-candidate 目录删除，
不再作为并行别名维护。算法全身份仍保存在本登记和新 manifest 中。

状态：**active candidate；尚未进入 `paper/main.tex`，也没有覆盖
`paper/Figures/ala2_66d/official_20260802/` 中的正式图。** 判断论文当前采用图时，
仍以 `paper/main.tex` 的 `\includegraphics` 为唯一真值。

本登记使用的仓库根目录为：

`/Users/wuzhiyou/IVM_for_MEP`

登记时 Git HEAD 为
`a48ba212abcb4123bece995e8c1fadc12a6d59e4`，但相关 `paper/ActiveExperiments/`
和 `python/` study 资产尚未被该 commit 跟踪；所以 commit SHA 只记录环境时点，不能
替代下文的逐文件 SHA-256。

除特别注明外，以下路径都相对于这个根目录。若本文件的人工描述与机器附件冲突，
优先级依次为：

1. same-stem `_manifest.json` 中登记的输入路径和 SHA-256；
2. same-stem `_source.csv` 中逐顶点的 `metric_source_file` 和
   `cost_source_file`；
3. 正式 campaign preregistration/report、member audit 和 NPZ；
4. 本 Markdown 的解释性文字。

这张图的 IVM **不是**旧的 `alpha=20`、Phase A34 + Phase B220 拼接结果。它是
一条从共同初值开始、没有 Phase A/B 切换的 single-run discrete IVM 轨迹。

## 2. 图、机器附件和当前哈希

| 角色 | 路径 | SHA-256 |
| --- | --- | --- |
| 登记 PDF | `paper/ActiveExperiments/alanine_66d/figures/active_candidates/alanine66d_discrete_ivm_cost_to_precision_1x2.pdf` | `3799bbeef793a98cc457188ce8d507e7e9e4b206041a03bdcf459acce588d255` |
| 逐顶点数据 | 同目录同 stem 的 `_source.csv` | `742723564379da1530ee5a6b1a9bd07e0d762f9e40456fd8ab45b9c01c3efe42` |
| 机器 manifest | 同目录同 stem 的 `_manifest.json` | `3edf1e29b6af34d290d3e20c8a6fe20bd20deb7ef1ca31a5f8c68a4dcb93ca18` |
| 唯一新图 plotter | `paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_discrete_single_cost_to_precision_muller_style.py` | `e7c1153f7ebb356d968c9015193637ad40e4124c55ecdf14c4b2f6c7ed12ce7f` |

PDF 是一页、180 mm x 72 mm 的 Matplotlib 3.7.2 PDF。`_source.csv` 有
14,026 个数据行：panel a 有 13,910 个 strict record vertices，panel b 有 116 个。
登记审计已重新计算 same-stem manifest 中 27 个 input hashes 以及 PDF/CSV 输出哈希，
当前全部匹配，未发现 manifest drift。
逐方法顶点数为：

| 方法 | panel a | panel b |
| --- | ---: | ---: |
| Discrete IVM | 18 | 5 |
| ODE-NEB | 3,741 | 21 |
| Static NEB | 4,500 | 25 |
| Original String | 4,500 | 25 |
| OBSM-FIRE | 1,128 | 29 |
| OBSM-BFGS | 23 | 11 |

`_source.csv` 每行保存 panel、method、metric、实际 metric value、最小 `N_F`、
source call、mesh 字段、dense subdivision、成本来源文件和 metric 来源文件。
它是从 PDF 上任一顶点回到 NPZ 的最短人工审计入口。

## 3. 最短完整 provenance 链

```text
registered PDF
├── same-stem _source.csv
│   └── 每个实际画出的 strict record vertex 与 source_file
├── same-stem _manifest.json
│   └── 输入/代码/PDF/CSV 哈希、坐标、first-passage 和运行身份
└── plot_ala_discrete_single_cost_to_precision_muller_style.py
    ├── new one_way_takeover result NPZ
    │   ├── member command.json
    │   ├── member audit + runner companion audit
    │   ├── campaign preregistration/report
    │   ├── postrun mechanism audit
    │   └── formal I_perp CTP audit
    ├── new complete+mesh dense-barrier NPZ + audit JSON
    │   └── postprocess_dense_barrier.py
    └── five frozen baseline raw/metrics/dense NPZ files
        └── legacy baseline runners/loaders
```

## 4. 共同物理体系、初值和计费

- 体系：真空 Alanine dipeptide，Amber14/OpenMM，22 个原子，66D。
- 路径：22 个 images，其中 20 个为可动内部 images、2 个为固定端点。
- topology：`python/alanine_ivm_clean_four_level_20260801/inputs/ala2_fixed.pdb`，
  SHA-256
  `7ffaf924e345b6fbbba06d0388a67d18f04c07729a2ab7d65ede9a61cd27eda2`。
- 新 single-run IVM 的 canonical common-initial 文件：
  `python/alanine_ivm_clean_four_level_20260801/inputs/common_initial_path_equidistant.npz`，
  文件 SHA-256
  `4a08066e1f0a89ad1d0ee6ecc442d7255d511217576d1ff65e9cb5e2164d891b`；
  `initial_path` 数组 SHA-256
  `cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e`。
- tangent：normalized central difference `x[i+1]-x[i-1]`。
- 新 IVM 的横轴读取结果 NPZ 中实际保存的
  `total_internal_joint_evaluations`，不是 outer 数，也不是事后用理论步数乘 20。
  这个字段只合计 20 个可动内部 images；两个固定端点的初始 evaluation 不在其中。
- 五条历史 baseline 沿用其冻结图表口径：generic baseline 用整个
  `counter_vectors.sum(axis=1)`，ODE-NEB 用 `20*count+2`，都包含两个固定端点各一次。
  因此当前图存在一个固定的 endpoint 计费口径差：新 IVM 比 baseline 的 all-image
  口径少 2。比如图中的 IVM `N_F=2150/911` 若统一成 all-image ledger，会变成
  `2152/913`；`38.8x` 和 `21.1x` 的一位小数显示不变。图轴现在使用中性的
  `Number of Forces`，但不能据此声称 endpoint convention 已完全统一。
- rejected trial、reparameterization direct solve/recovery 和真实 cache miss 均由 runner
  ledger 计费；固定端点不作为优化内部 image 成本重复计费。
- 新 IVM 的 7,544 次 dense diagnostic 是事后分析，不参与算法状态选择，明确排除在
  optimization `N_F` 之外。
- Baseline 的离线 dense diagnostics 同样不进入各自优化 `N_F`。

正式 preregistration 锁定单线程环境：

```text
OPENMM_CPU_THREADS=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
```

## 5. 新 single-run discrete IVM 的算法身份

图例中的蓝线只简写为 `IVM`；它实际使用 `one_way_takeover` arm，完整固定身份为：

| 项目 | 固定值 |
| --- | --- |
| 运行结构 | 从共同初值开始的一条连续 single run；无 Phase A/B |
| `alpha` | 固定 15，全程不变 |
| post-selection proposal multiplier | `eta=1.1` |
| frozen subproblem | `discrete_b040`：直接参数化 20 x 66 个离散内部点坐标 |
| frozen-path tangent/projector | 每个 fixed subproblem 开始时冻结；subproblem 内不重参数化 |
| inner stop | `R/R0 <= 0.8 OR evaluated accepted steps == 10` |
| absolute KKT stop | 关闭；`kkt_atol=0` |
| L-BFGS history | `m=7` |
| direction mechanism | frozen FFrbk `H0` 与同一 operator 初始化的 L-BFGS |
| selector | 每个 fixed subproblem 内首次 strict L-BFGS 胜出后 one-way preference latch |
| invalid-LBFGS safeguard | 保留 H0 failover；本次实际 0 次 |
| `eta` 位置 | 原 selector 选中方向之后、原 dynamic cap 重新施加之前 |
| energy/objective backtracking | 无；没有加入本轮已放弃的 energy fallback |
| outer reparameterization | 现有 `state_curve/per_image`，从第一状态开始 |
| per-image budget | 254；保存 254 个 trajectory records |
| physical stop | `evaluation_budget`，不是数值收敛声明 |

固定子问题的目标仍是 frozen outer reference 上的离散 proximal 问题：

```text
Phi(D) = sum_i [ V(phi_i + D_i) + ||D_i||^2 / (2 alpha) ]
subject to D_i perpendicular to frozen tau_i.
```

FFrbk bank 在一个 fixed subproblem 中冻结；L-BFGS 的 secant history 只接收已评价并
提交的同一 frozen problem 状态。`eta=1.1` 只放大已选方向，之后仍受原 dynamic
active-step cap 保护。本次正式运行实际没有发生 `eta` cap hit。

### 5.1 机制审计摘要

| 审计量 | one-way run |
| --- | ---: |
| fixed subproblems | 22 |
| `R/R0 <= 0.8` stops | 22 |
| 10-step cap stops | 0 |
| H0 selections | 22 |
| L-BFGS selections | 49 |
| strict-L takeover triggers | 19 |
| post-latch forced-L selections | 30 |
| empty-history ties | 22 |
| H0 safety failovers | 0 |
| `eta` cap hits | 0 |
| final optimization `N_F` | 3,389 |

同一 campaign 还运行了 `forced_operator_lbfgs_selection` arm。postrun audit 证明它与
`one_way_takeover` 的 63 个安全科学轨迹字段、`stored_paths`、`N_F`、`I_perp`、
mesh/reparameterization 轨迹逐字节相同。因此本图只画 one-way 一条蓝线，不把
forced arm 冒充第二个独立结果。

## 6. 新实验的可执行代码

以下是 preregistration 在运行前锁定、运行后重新验证的核心代码：

| 代码角色 | 路径 | SHA-256 |
| --- | --- | --- |
| 两 arm campaign grid | `python/alanine_ivm_clean_four_level_20260801/src/run_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_grid.py` | `45b0d0bbc5abc0deba61e20b09ad6d5391cb350a532cfd6159cf1cc72170adf5` |
| 单 member wrapper/validator | `python/alanine_ivm_clean_four_level_20260801/src/run_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0.py` | `612aad99b0edc6cd8352de2bee9f73a05af7cbfc776048da828381653eaeb304` |
| one-way/forced selector runtime | `python/alanine_ivm_clean_four_level_20260801/src/run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py` | `62c6cf784795984c8104b4ba28fe2b0e15d657593648734d8f4a36ae03edfffc` |
| post-selection `eta` runtime | `python/alanine_ivm_clean_four_level_20260801/src/run_ivm_v9_discrete_selected_eta_shim.py` | `a2f071ea787a008ed001fd806b8952f84d8db90335b544308478f3f6ae2facf9` |
| direct discrete fixed solver | `python/alanine_ivm_clean_four_level_20260801/src/frozen_discrete_b040.py` | `8384a84666af463991df2e8d8a344779ae675feaddc73c12bcc82ff6ba324e22` |
| production outer runner/reparameterization state machine | `python/alanine_ivm_clean_four_level_20260801/src/run_ivm_v9_spring_projected_production_20260730.py` | `9abd6e96bb9e6cead380a28c849fd11c15c5fe9a8854d77fe41a991240b0510f` |
| common Ala evaluator/metrics/force field | `python/alanine_ivm_clean_four_level_20260801/src/ala_core.py` | `e3d9b2d472d8b9d7e0e03289dcf6a87bdff10cb8fc296bdfda911798799377bd` |
| canonical command helper | `python/alanine_ivm_clean_four_level_20260801/src/run_stage4_uniform_optimized.py` | `990be5520093af674bc44061c0c38af9169a1e207f06c33f7bea9d53ac2d7e02` |
| formal Iperp CTP analyzer | `python/alanine_ivm_clean_four_level_20260801/src/analyze_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_iperp_ctp.py` | `9f3356e77d5122c02ef905d542f1357f581643f47aeedb7292eec85625d749b3` |
| postrun mechanism/provenance auditor | `python/alanine_ivm_clean_four_level_20260801/src/audit_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_postrun.py` | `1909b17d0fa34607ed6fcb4ea94240087ba292261bc73e75e85cb103a92e0d96` |

关键测试在同一 study 的 `tests/` 下。preregistration 的 `run_source_manifest` 保存了
runtime、frozen solver、campaign 和测试的运行前 SHA；postrun audit 再验证所有登记
source 和 artifact hash。

## 7. 新实验结果、命令和审计链

### 7.1 代表结果

| 角色 | 路径 | SHA-256 |
| --- | --- | --- |
| one-way result NPZ | `python/alanine_ivm_clean_four_level_20260801/runs/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1/members/ivm_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_one_way_takeover_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1_central_state_ivm_cfg1804ab58ab75_budget254.npz` | `1c81ad6fe876a0eb94f2fc55d5d2567737face95ff52b6fe1bdfffc536d16787` |
| member scientific audit | `python/alanine_ivm_clean_four_level_20260801/runs/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1/audits/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_one_way_takeover_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1.json` | `bd1caa6b744ede59e95b849eef8bf92a80ca321e849754c28d5f726e1147969f` |
| runner companion counter audit | same result stem 的 `.audit.json` | `d71bf5d14e7f2d070a3da4e70e85c9276bceb43920a9598de81b789ff2b63534` |
| exact member command JSON | `python/alanine_ivm_clean_four_level_20260801/logs/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1/members/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_one_way_takeover_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1/command.json` | `a970b975338c14b8f39cb6b131ccb80fe44f36c7b3801ea18101cca0f54ef93d` |
| launcher command JSON | 相邻 `logs/.../launcher/.../command.json` | `955a5e78dd0cc785a8321a66c3da9d9377e992975634c27dbd7cd2840cb9a94e` |
| forced identity result（不单独画） | 对应 `forced_operator_lbfgs_selection...cfgdbf51ea22ca4_budget254.npz` | `5d8169d1d16dab2836bef43affcaaec2dabf539a8e0f992c9050cd909faf4050` |

one-way 结果的 `configuration_sha256` 为
`1804ab58ab7551de2573320a04f07bb369872f5d333d50610d49d4ff7c200df6`；
注册 scientific command 的 compact SHA-256 为
`b4e8b7bb9493edda11c10b3ef384f988d5072fe8ec5e20acd72043131d05f965`。

### 7.2 Campaign 级别证据

| 角色 | 路径 | SHA-256 |
| --- | --- | --- |
| preregistration | `python/alanine_ivm_clean_four_level_20260801/reports/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_N254_r1_1787262084320785000_preregistered.json` | `67dd26b93f216a2d7f429a1b31710bdf1b933c24e10f17779e2499621705fa4e` |
| complete campaign report | 同 stem `.json` | `c64c471d612a22161ff1308bd969f83588b5de4a2aa67e62b4e4b848e96514ad` |
| formal Iperp CTP audit | 同 stem `_iperp_ctp_readonly_v1_1787262442236424000.json` | `e9f5008381617fa98b04fabeb05cbc8b347e809dce9cdbd0901b7a0fc5a3e80d` |
| postrun mechanism/provenance audit | 同 stem `_postrun_mechanism_provenance_readonly_v1.json` | `28b3c12692ba6ac21576dd193677916b77355370a41a009ba9e33b0a615ed538` |

### 7.3 精确 member 命令

下面命令与 `command.json` 的 token 序列一致；`command.json` 和 preregistration
仍是避免换行/转义歧义的机器真值。

```bash
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  /Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801/src/run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py \
  --root /Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801 \
  --initial-path-file inputs/common_initial_path_equidistant.npz \
  --output-dir runs/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1/members \
  --run-label stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_one_way_takeover_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1 \
  --solver state_ivm --alpha 15 --alpha-mode fixed --alpha-min 0.01 --alpha-max inf \
  --max-step 1 --history 7 --stop-iperp -1 --budget 254 --save-all-paths \
  --state-globalization iperp_monotone \
  --state-trust-fraction 2 --state-polish-trust-fraction 2 \
  --state-polish-alpha-factor 0.5 --inner-calls 15 --plbfgs-kkt-rtol 0.8 \
  --reparam-mode state_curve --reparam-every 1 --state-curve-direct-solve \
  --state-curve-direct-trial-calls 15 \
  --state-curve-direct-objective-acceptance per_image \
  --state-curve-immediate-block-relock --state-curve-stall-free-polish \
  --state-curve-stall-free-polish-count 1 --reparam-ready-relative-iperp 2 \
  --reparam-cv-trigger 0.001 --reparam-strict-cv-trigger 0.001 \
  --reparam-retry-reduction 0.8 --reparam-recovery-calls 320 \
  --reparam-recovery-inner-calls 15 --reparam-recovery-merge bicriteria \
  --reparam-recovery-iperp-rtol 0 --reparam-equidistance-contraction 0.2 \
  --stop-complete-iperp -1 --frozen-subproblem discrete_b040 \
  --discrete-inner-max-steps 10 --discrete-stop-mode r08c10 \
  --discrete-residual-ratio 0.8 --discrete-history 7 \
  --discrete-step-multiplier-eta 1.1 --plbfgs-kkt-atol 0 \
  --discrete-ffrbk-selector-mode one_way_takeover
```

正式结果已经存在。不得用同一个 output path/replicate 静默覆盖；要重跑时使用新的
replicate/campaign id，或先明确登记替换决策。

## 8. 图中 IVM 状态如何从 254 条轨迹选出

正式 eligibility 规则为：

```text
(initial state OR reparam_committed)
AND segment_length_cv <= 0.0065
AND segment_length_ratio <= 1.03
```

23 个 complete states 中有 18 个通过 mesh gate。图的 IVM 两个 panel 都从同一批
18 个路径出发；panel b 随后读取这些路径的独立 dense metric。

| source call | `N_F` | `I_perp` | dense `Delta E` (kcal/mol) |
| ---: | ---: | ---: | ---: |
| 1 | 20 | 776.2298999129 | 44.5600075748 |
| 4 | 80 | 0.461460850338 | 3.49283686592 |
| 7 | 140 | 0.108668523525 | 3.20380859564 |
| 9 | 180 | 0.0643845572517 | 3.06395818310 |
| 50 | 911 | 0.00143178138991 | 2.45759824353 |
| 56 | 1,035 | 0.000881570329954 | 2.48358741558 |
| 62 | 1,157 | 0.000630394465188 | 2.50376235513 |
| 86 | 1,460 | 0.000312932616941 | 2.50721192668 |
| 96 | 1,569 | 0.000273092591355 | 2.51525585996 |
| 102 | 1,688 | 0.000244656303676 | 2.51094657248 |
| 108 | 1,820 | 0.000221912637066 | 2.51231690935 |
| 114 | 1,941 | 0.000212029263860 | 2.51372912589 |
| 134 | 2,150 | 0.000190958695731 | 2.52749979360 |
| 140 | 2,282 | 0.000183941682136 | 2.53188076228 |
| 166 | 2,519 | 0.000165588678671 | 2.53743948546 |
| 190 | 2,781 | 0.000157125504145 | 2.54350669501 |
| 214 | 3,045 | 0.000151779078170 | 2.54736525094 |
| 254 | 3,389 | 0.000146586064816 | 2.54809950554 |

所有 18 个 `I_perp` 都是 strict running-low records，因此全部进入 panel a。dense
barrier 在 call 50 后回升，所以 panel b 的 IVM cost-to-lower-barrier frontier 只保留
calls `1,4,7,9,50`。这不是丢数据：18 条 dense values 都保存在 dense NPZ、manifest
和上表中；PDF 只画满足 record 定义的 5 个顶点。

## 9. 新 IVM densified barrier 的来源和 7,544 次诊断计数

| 角色 | 路径 | SHA-256 |
| --- | --- | --- |
| dense NPZ | `python/alanine_ivm_clean_four_level_20260801/runs/diagnostics/dense_barrier_stage4_discrete_single_one_way_eta1p1_r08c10_kkt0_a15_complete_mesh.npz` | `da2eb76b4c40347212914e493abe0e440617a36052f9e64609131af39c1e342a` |
| dense audit | 同 stem `.audit.json` | `b3c9b9b12900410cd1e5f3809ee47f0f75f8e35ecdb35494e13e7a286da03c46` |
| postprocessor | `python/alanine_ivm_clean_four_level_20260801/src/postprocess_dense_barrier.py` | `18925061b147f2adfd1d34705a91870385eeb8e152fe582b618bdfa1444fe38b` |

Postprocessor 对 22-image 路径的 21 个 Cartesian segments 线性加密并重新调用
OpenMM。先计算 10 subdivisions，再计算 20；若差值大于 0.05 kcal/mol 才继续
40/80。本次 18 条路径全部在 20 subdivisions 停止，最大 10/20 差约
0.003166 kcal/mol。图读取 `adaptive_barrier_kcal`；本次它与 `barrier_20_kcal`
逐点相同。结果 NPZ 中运行时的 sparse `barrier_kcal` **没有用于 panel b**。

20 subdivisions 对一条 22-image 路径产生：

```text
21 segments * 20 samples per segment + final endpoint = 421 coordinates.
```

10-subdivision 网格是 20-subdivision 网格的子集，cache 后不额外重复计费。18 条路径
共享两个固定端点，因此第一条 421 次，后续 17 条各 `421-2=419` 次：

```text
421 + 17 * 419 = 7,544 OpenMM joint evaluations.
```

Audit 的 backend raw count、evaluator cache misses 和逐 record ledger 都等于 7,544。
这些调用只证明事后 dense metric，不进入图的 optimization `N_F`。

下面是由已冻结 source/result/output/calls 和 adaptive 参数恢复出的可复现重放命令。
Dense artifact 没有单独保存历史 `command.json`，因此 `--workers 1` 是确定性重放选择，
不是可由现存 artifact 证明的历史 worker 数；worker 数不改变科学输入、subdivision
规则或已审计的 7,544 次物理调用。

```bash
cd /Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  src/postprocess_dense_barrier.py \
  --root /Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801 \
  --result /Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801/runs/stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1/members/ivm_stage4_ala_discrete_single_ffrbk_takeover_matched_kkt0_v1_one_way_takeover_eta1p1_r08c10_kkt0_a15_N254_discrete_b040_r1_central_state_ivm_cfg1804ab58ab75_budget254.npz \
  --output-label stage4_discrete_single_one_way_eta1p1_r08c10_kkt0_a15_complete_mesh \
  --workers 1 --calls-key stored_path_calls --paths-key stored_paths \
  --calls 1,4,7,9,50,56,62,86,96,102,108,114,134,140,166,190,214,254
```

重算时应改用新的 output label，先比较新旧 hash/audit；不要静默覆盖已登记 dense
artifact。

## 10. Plotter、读取逻辑和样式依赖

| 代码角色 | 路径 | SHA-256 |
| --- | --- | --- |
| 本图唯一入口 | `paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_discrete_single_cost_to_precision_muller_style.py` | `e7c1153f7ebb356d968c9015193637ad40e4124c55ecdf14c4b2f6c7ed12ce7f` |
| Müller-style 函数/色板底座 | `paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_cost_to_precision_muller_style.py` | `b0b7815a159092d72b12cf3ebd2f2b63cab895a20b251c61e035e1e2fef6350b` |
| baseline loader import dependency | `python/alanine_ivm_clean_four_level_20260801/src/plot_all_methods_linear_speedup_v5.py` | `e52e361766bf866c8c677e2c77cee2749afb6dac9524dc2f3aa68f33699077c8` |
| legacy `MethodSeries`/baseline loader | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/src/plot_ala_integer20_all_methods.py` | `599f8bede9608a68f78947b7446b068e40186e673a50e67d9478cd795ed5d260` |
| imported legacy-module adapter | `python/alanine_ivm_clean_four_level_20260801/src/plot_stage4_v4_pareto_all_methods.py` | `50151768a2ab6f13f01c7b40a328a28333f37fe3c50e83a22d91521e98bff9a7` |
| transitive `cost_to_reach` source | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/src/plot_v49_formal_ala_1x3.py` | `4af8feee0843206f17fc6f4b1b19eb52743719a4f553821a8a041ac9d06626dd` |
| transitive legacy import dependency | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/src/plot_v67_continuation_figures.py` | `84903ddb16e5b70dda17589dadf332dd89e3b2053c9e475ac4cf507b32032149` |

新 plotter 只复用旧 plotter 的字体、颜色、轴、record selector 和 loader class。它有
独立的 baseline-only loader，并直接从新 single-run result 构造 IVM `MethodSeries`；
它不会打开旧的 `stage4_uniform_v3_a20_phaseA34_phaseB220.npz`、旧 Pareto archive 或
旧 IVM dense files。manifest 中 `old_phase_a_b_ivm_used=false` 指的是数值输入；旧代码
文件作为 import/style 依赖出现不代表旧 IVM 数据进入了图。

当前 same-stem manifest 尚未列出上表最后两个 transitive imports。前者实际提供
first-passage 使用的 `cost_to_reach`，后者是 legacy module 成功导入所需依赖；本 MD
补登记其当前 SHA，直到 plotter/manifest 将它们纳入机器 source manifest 为止。

## 11. 两个 panel 和箭头的精确定义

### Panel a

- metric：formal complete+mesh 状态上的实际 `I_perp`。
- x label：`I_perp`；反向 log，`1e-1 -> 1e-4`。
- y label：`Number of Forces`；linear，`0 -> 110k`。
- pairwise target：`I_perp <= 2.1e-4`。
- new IVM first passage：2,150；ODE-NEB：83,522；比值
  `38.8474418605...`，图示 `38.8x`。

### Panel b

- metric：实际 densified energy barrier `Delta E`，不是
  `|Delta E-Delta E*|`，没有 tolerance band。
- x label：只显示 `Delta E (kcal mol^-1)`，不再写 `Densified energy barrier`；
  反向 log，`3.0 -> 2.4 kcal/mol`。
- y label：`Number of Forces`；linear，`0 -> 45k`；更高 baseline 尾部由坐标范围裁切。
- pairwise target：`Delta E <= 2.4721831748734475 kcal/mol`。
- new IVM first passage：911；ODE-NEB：19,222；比值
  `21.0998902305...`，图示 `21.1x`。

箭头旁只显示 `38.8x`/`21.1x`，但机器 manifest 明确记录它们是
ODE-NEB/IVM 的 metric-specific marginal first-passage ratio。两个数字不是同一个
checkpoint 的 joint test，也不是“相对所有五个 baseline 的最快倍数”。

### Cost-to-precision record 构造

对每个方法和每个 metric 独立执行：

1. 按实际 `N_F` stable ascending 排序；
2. 同一 `N_F` 若有多条观测，只保留 metric 最低者；
3. 仅在 metric 严格刷新 running-lowest 时保留实际 checkpoint；
4. 按 `N_F` 顺序用普通线段连接实测 record vertices。

没有 threshold-grid 插值、线段交点计费、外推或阶梯 drawstyle。线只是视觉引导；
first passage 只从实际保存的顶点读取。

## 12. 五个 baseline 的冻结数值输入

新图保留旧 1x2 candidate 的五条 long-budget baseline，不重跑、不重新调参。

| 方法/作用 | 路径 | SHA-256 |
| --- | --- | --- |
| ODE raw/path/count | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/formal_figure_inputs/ala_0to1_N22_k50_alpha0.0001_ode_results.npz` | `935561709ecbdfe87af31b78b45335a66545356b3e62f331a4a9fbb8125d9a09` |
| ODE panel-a metrics | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/formal_figure_inputs/metrics_ala_0to1_N22_k50_alpha0.0001_ode_results.npz` | `b2535ddc90bd21142cf5bd1a7bd73428c3856f8fe559cdcfd030df9f3312241c` |
| ODE dense | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/runs/diagnostics/dense_barrier_ala_ode_neb_k50_logsample_dense.npz` | `b85baf1c0a6e04535771768f112e6737507a4a9ee4e29f0ece4bea1971ad9dab` |
| Static raw | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/runs/neb/neb_publication90k20260802r2_central_k30_alpha0p013_static_budget4500.npz` | `af6228a27324baf7a61ae34d55e7d3c0e7cd66933c6ebd14641e1b82eb0b6bf8` |
| Static dense | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/runs/diagnostics/dense_barrier_ala_static_k30_dt0p013_publication90k.npz` | `4b4f021d15e6df846ecf0d504196c5a26bba64095609408a597e2292d81d48f7` |
| Original raw | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/runs/publication_long90k_20260802/publication_long90k_20260802_publication90k20260802r2_original_nolambda_dt0p014_r1_cfg587b411dd523_budget4500.npz` | `e161d010ff662ae7ad12b2552ae02e152272ed124b4da2e05254c12d011862bd` |
| Original dense | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/runs/diagnostics/dense_barrier_ala_original_dt0p014_publication90k.npz` | `8f8301d8cd3a994fd1005e834fd6196770b5ed5e5680e541d07bb1aa10ffdf07` |
| FIRE raw | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/runs/publication_long90k_20260802/publication_long90k_20260802_publication90k20260802r3_fire_lam0p85_inner3_r1_nocap_cfg75c2b3203822_budget4510.npz` | `33d3963eb17887e7a2779ab1533ee103547366a10517cb19c3907ef9396354bf` |
| FIRE dense | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/runs/diagnostics/dense_barrier_ala_obsm_fire_lam0p85_inner3_dt0p1_publication90k.npz` | `d9ff02b2da9bf6131801839f09cc8c42c2f3ee4621e82fc2eb4de391b708ae2a` |
| BFGS raw | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/runs/publication_long90k_20260802/publication_long90k_20260802_publication90k_robust_20260802r2_bfgs_lam0p3_inner1_r1_nocap_cfg9c7658b737c3_budget8000.npz` | `d8033d1de78841fa00dcfceb41118465c32465ec66957d818620487369483621` |
| BFGS dense | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/runs/diagnostics/dense_barrier_ala_obsm_bfgs_lam0p3_robust_publication90k.npz` | `ec5518556cc15a38784c5e1195cfb2593d682c003cdb6aac1edc3c45e788e1c4` |

配置摘要：ODE-NEB 使用冻结 `k=50` 高精度轨迹；Static NEB 使用 `k=30,
dt=0.013`；Original String 使用 `dt=0.014`；OBSM-FIRE 使用 `lambda=0.85`、
inner 3、`dt=0.1`；OBSM-BFGS 使用 `lambda=0.3`、inner 1、bonded `H0` 和
cross-outer carry。完整 baseline 调参范围和限制见
`paper/ActiveExperiments/alanine_66d/README_FIRST.md` 以及旧 Phase-A/B 图的专用
`FIGURE_COST_TO_PRECISION_1X2_ALL_METHODS_PROVENANCE.md`。

对应实验代码：

| 方法 | runner | 当前文件 SHA-256 |
| --- | --- | --- |
| Original String | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/src/run_string_family.py` | `8473284900c7ed8a400318d86b8dccb6c1fe9c677866dd62de5dd02bb8e8a393` |
| Static/ODE NEB | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/src/run_neb_competitor.py` | `18cce0cbc6d6d50342250e8276b7fc1bc944ebdb805f383bda76ec561b34d82a` |
| OBSM-FIRE/BFGS | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/src/run_obsm.py` | `999dbce937a04840f2c50f58e8a82c07124e98bf8ee5844c26d07253d3b066fb` |
| 共用物理实现 | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/src/ala_core.py` | `e3d9b2d472d8b9d7e0e03289dcf6a87bdff10cb8fc296bdfda911798799377bd` |

这些是当前 executable working source 的哈希；冻结结果 NPZ、same-stem audit/config
和图 manifest 中的 artifact hash 才是 baseline 结果身份的最高优先级证据。

## 13. 一条命令重画 PDF、CSV 和 manifest

从仓库根执行：

```bash
cd /Users/wuzhiyou/IVM_for_MEP
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/ala_discrete_single_plot_mpl \
XDG_CACHE_HOME=/private/tmp/ala_discrete_single_plot_cache \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_discrete_single_cost_to_precision_muller_style.py \
--overwrite
```

该命令同时更新登记 PDF、same-stem `_source.csv` 和 `_manifest.json`。Plotter 在写图
前会按硬编码 SHA fail-close 验证 8 项新实验输入：one-way/forced result、dense
NPZ/audit、campaign report、preregistration、formal `I_perp` CTP 和 postrun audit；
并重新计算 complete+mesh 索引、dense calls 对齐、forced-arm safe-field identity 和
两个 first-passage 数值。

Baseline NPZ、common initial/topology、dense postprocessor 和绘图/import 依赖当前不会
在重画前对旧 SHA fail-close；plotter 会把它们重画时的当前 SHA 写进新 manifest。
因此接受新输出前仍必须把 manifest 的 27 个 input hashes 与本登记/上一份 manifest
逐项比较。不能把“成功生成新 manifest”本身理解为所有依赖都未漂移。

Matplotlib PDF 包含 `CreationDate`，即使视觉和数据不变，重新导出也会改变 PDF 的
byte-level SHA。任何有意重画后都必须重新登记 PDF/manifest/plotter 哈希；不能保留
本文件中的旧哈希并声称 byte-identical。

## 14. 最小核验顺序

1. 对第 2 节四个登记文件运行 `shasum -a 256`。
2. 打开 same-stem manifest，先检查：
   `single_run_no_phase_a_b=true`、`old_phase_a_b_ivm_used=false`、
   `sparse_result_barrier_kcal_used=false` 和 `y_range_NF=[0,45000]`。
3. 在 `_source.csv` 里按 `method=ivm,panel=a/b` 追到 one-way result/dense NPZ。
4. 对照 member audit 的 `scientific_command_sha256`、`configuration_sha256`、
   `initial_path_sha256` 和 result SHA。
5. 对照 dense audit 的 18 calls、7,544 raw queries、20-subdivision histogram 和
   `counts_excluded_from_optimization_axis=true`。
6. 对照 campaign postrun audit，确认 one-way/forced safe trajectories identical，
   selector/eta 没有额外 physical evaluations。
7. 用 Poppler 渲染 PDF，确认图例蓝线为 `IVM`、顶部说明小字已删除、两幅图纵轴均为
   `Number of Forces`、右图横轴只显示 `Delta E` 和单位、右图上限 45k，并确认两个
   箭头只显示 `38.8x`/`21.1x` 且标签位于箭头中点右侧。

## 15. 已知限制和禁止表述

- 这张图是 active candidate，不是当前论文正式图。
- 新 IVM 横轴排除两个固定端点的初始 evaluation，冻结 baseline 横轴包含它们；这是
  固定 `+2` 的 convention 差，不影响图示一位小数倍数，但意味着 endpoint 计费没有
  严格统一。
- 两个箭头是分别在两个 metric 上回看历史得到的 marginal first passage；不能称为
  同一 checkpoint 的 joint speedup。
- 新 IVM 在 `N_F=911` 达到很低的 dense barrier，之后 barrier 回升；panel b 的
  running-low frontier 不展示回升段。不能把 `21.1x` 解释为最终路径 barrier 一直
  保持在该值。
- Frozen ODE artifact 的 `images_seq[0]` 与当前 common initial 的最大坐标差是
  `0.06150258892739646 Angstrom`。图可称为对冻结正式 baseline 的比较，但不能称为
  六方法严格 exact-same-IDPP-artifact comparison。
- forced arm 与 one-way arm 的轨迹 identical，只能作为机制 identity 证据，不能画成
  两条独立成功曲线。
- 这张图没有使用旧 Phase A/B IVM 数据，也没有修改旧 official PDF、`main.tex`、
  baseline raw NPZ 或 baseline dense NPZ。
- Dense 后处理没有独立历史 `command.json`，不能把重放命令中的 `--workers 1` 当成
  已证明的历史 worker 数。
- 当前 manifest 漏列两个 legacy transitive imports；本登记第 10 节用逐文件 SHA
  补足。仓库当前 Git commit 也不能单独标识这些实验资产，因为相关 study/paper 树
  尚未被 commit 跟踪，必须依赖这里和机器 manifest 的逐文件 SHA。

## 16. 后续更新规则

任何下列变化都要求新的 candidate stem 或明确的替换登记：

- 改 `alpha`、`eta`、`r`、max evaluated steps、history 或 selector；
- 改 reparameterization、mesh gate 或 complete-state 定义；
- 改 dense calls、subdivisions/tolerance 或把 dense 成本计入优化横轴；
- 更换任一 baseline raw/metrics/dense artifact；
- 改 cost-to-precision record 定义、轴范围、箭头 target 或标注语义；
- 把这张 candidate 提升到 official 或写入 `paper/main.tex`。

更新顺序应为：先冻结实验/诊断 artifact 和 audit，再重画 PDF/CSV/manifest，最后更新
本 Markdown 的哈希与解释。不得先改 PDF 再从视觉倒推其数据来源。
