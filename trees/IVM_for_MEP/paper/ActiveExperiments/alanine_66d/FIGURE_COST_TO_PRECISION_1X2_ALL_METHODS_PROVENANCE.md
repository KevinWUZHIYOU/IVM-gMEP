# Alanine 66D 六方法 1x2 cost-to-precision 图溯源

登记日期：2026-08-21（Asia/Hong_Kong）

## 1. 登记对象与状态

本文件只登记下面这一张图：

`paper/ActiveExperiments/alanine_66d/figures/active_candidates/alanine66d_stage4_v4_cost_to_precision_1x2_all_methods_muller_style.pdf`

状态：**active candidate；尚未进入 `paper/main.tex`，也没有覆盖
`paper/Figures/ala2_66d/official_20260802/` 中的正式图。** 判断论文当前采用图时，
仍以 `paper/main.tex` 的 `\includegraphics` 为唯一真值。

本登记使用的仓库根目录为：

`/Users/wuzhiyou/IVM_for_MEP`

除特别注明外，以下路径都相对于这个根目录。

## 2. 图与机器附件

| 角色 | 文件 | SHA-256 |
| --- | --- | --- |
| 登记 PDF | `paper/ActiveExperiments/alanine_66d/figures/active_candidates/alanine66d_stage4_v4_cost_to_precision_1x2_all_methods_muller_style.pdf` | `c486a55e3b4dae2267b2de20dae4db0528ddc780d9a6a2072a399e190ee830e3` |
| 逐点数据 | 同目录同 stem 的 `_source.csv` | `3b692549012e675a8dfc638a731023a9a4121e886d211e187ed5ae54a6b39cd8` |
| 机器 manifest | 同目录同 stem 的 `_manifest.json` | `724c924c59310ef71b5eb1864794ce8bf4d7b5ff5680d65a6267741ae0c3fec6` |
| 唯一 candidate plotter | `paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_cost_to_precision_muller_style.py` | `b0b7815a159092d72b12cf3ebd2f2b63cab895a20b251c61e035e1e2fef6350b` |

`_source.csv` 有 14,053 个数据行，只含 panel a/b 的实测 record vertices。
每行包含 `panel`、`method`、`metric_value`、`NF`、`source_stage` 和
`source_file`。`source_stage` 已能区分：

- IVM 的 combined raw prefix 与 Pareto accepted tail；
- IVM 的 dense preactivation prefix 与 dense-screen accepted tail；
- ODE panel a 的独立 metrics NPZ；
- 其余方法的 raw result 与 dense diagnostic。

## 3. 最短完整 provenance 链

```text
registered PDF
├── same-stem _source.csv       # 每个实际画出的 record vertex
├── same-stem _manifest.json    # 输入/代码/PDF/CSV 哈希与数值审计
└── plot_ala_cost_to_precision_muller_style.py
    └── plot_all_methods_linear_speedup_v5.py
        ├── plot_stage4_v4_pareto_all_methods.py   # IVM 四路拼接与计费
        └── plot_ala_integer20_all_methods.py      # baseline/ODE loader 与 NF 映射
            └── raw/metrics/dense NPZ files
```

实际绘图代码依赖如下：

| 代码角色 | 路径 | SHA-256 |
| --- | --- | --- |
| candidate 入口 | `paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_cost_to_precision_muller_style.py` | `b0b7815a159092d72b12cf3ebd2f2b63cab895a20b251c61e035e1e2fef6350b` |
| 六方法 loader | `python/alanine_ivm_clean_four_level_20260801/src/plot_all_methods_linear_speedup_v5.py` | `e52e361766bf866c8c677e2c77cee2749afb6dac9524dc2f3aa68f33699077c8` |
| IVM Pareto loader | `python/alanine_ivm_clean_four_level_20260801/src/plot_stage4_v4_pareto_all_methods.py` | `50151768a2ab6f13f01c7b40a328a28333f37fe3c50e83a22d91521e98bff9a7` |
| baseline/ODE loader | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/src/plot_ala_integer20_all_methods.py` | `599f8bede9608a68f78947b7446b068e40186e673a50e67d9478cd795ed5d260` |

关键代码入口：candidate plotter 的 `metric_history()` 读取 metric，
`pareto_record_points()` 构造 strict record vertices，
`pairwise_comparison_record()` 计算 first passage，`draw_figure()` 画线和箭头，
`source_rows()`、`input_manifest()` 与 `initial_path_audit()` 生成逐点和机器溯源附件。
这些函数的位置以本文件登记的 plotter 哈希为准。

## 4. 共同物理问题与计费口径

- 体系：真空 Alanine dipeptide，Amber14/OpenMM，22 个原子，66D。
- 离散路径：22 个 images，其中 20 个为可动内部 images。
- topology：`python/alanine_ivm_clean_four_level_20260801/inputs/ala2_fixed.pdb`，
  SHA-256 `7ffaf924e345b6fbbba06d0388a67d18f04c07729a2ab7d65ede9a61cd27eda2`。
- 共同初值文件：
  `python/alanine_ivm_clean_four_level_20260801/inputs/common_initial_path_equidistant.npz`，
  文件 SHA-256 `4a08066e1f0a89ad1d0ee6ecc442d7255d511217576d1ff65e9cb5e2164d891b`；
  其中 `initial_path` 数组 SHA-256
  `cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e`。
- 所有方法使用 normalized central-difference tangent。
- generic baseline 的 `N_F` 是 22 个 image counter 的和；被拒绝/隐藏的实际
  energy-force 查询仍在 raw ledger 中计费。
- ODE 历史使用 `N_F = 20 * count + 2`。
- baseline 的离线 densified-barrier diagnostic 调用不进入优化 `N_F`。
- IVM 的 reference-free Pareto screen 是算法选择的一部分，其 2,301 个
  force-equivalent evaluations 全部计入；最终是 2,507 optimization + 2,301
  screen = `N_F=4,808`。

共同设置的最高层说明是 `paper/ActiveExperiments/alanine_66d/README_FIRST.md`。

## 5. 两个 panel 的精确定义

### Panel a

- metric value 来自实测 raw `I_perp` checkpoint；随后只保留 strict
  running-lowest record vertices，不对 raw 值做平滑或插值。
- x：反向 log，显示 `1e-1 -> 1e-4`。
- y：linear whole-path `N_F`，显示 0--110k。
- 比较目标：`I_perp <= 2.1e-4`。

### Panel b

- metric：实际 densified energy barrier `Delta E`，单位 kcal/mol。
- 它不是 `|Delta E-Delta E*|`，也没有 tolerance band。
- x：反向 log，显示 `3.0 -> 2.4`。
- y：linear whole-path `N_F`，显示 0--65k。
- 比较目标：`Delta E <= 2.4721831748734475`；该值是冻结 ODE 历史达到的
  最低实际 densified barrier。

### Cost-to-precision record 构造

对每个方法和每个 metric 独立执行：

1. 按 `N_F` stable ascending 排序；
2. 同一个 `N_F` 若有多个观测，只保留 metric 最低的观测；
3. 从低成本到高成本扫描，只在 metric **严格降低** 时保留实际 checkpoint；
4. 按观测时间/成本顺序连接这些 record vertices。

所以每个顶点都是“达到该实际精度时已观察到的最小 `N_F`”。线只作视觉引导；
成本计算不使用线性插值、阈值网格插值或外推。显示窗口外的 record 点仍保存在 CSV。

逐方法 record 数：

| 方法 | panel a | panel b |
| --- | ---: | ---: |
| IVM | 37 | 13 |
| ODE-NEB | 3,741 | 21 |
| Original String | 4,500 | 25 |
| Static NEB | 4,500 | 25 |
| OBSM-FIRE | 1,128 | 29 |
| OBSM-BFGS | 23 | 11 |

## 6. 箭头数字及其限制

| metric-specific target | IVM first passage | ODE first passage | ODE/IVM |
| --- | ---: | ---: | ---: |
| `I_perp <= 2.1e-4` | 4,808 | 83,522 | 17.371464...（图示 17.4x） |
| `Delta E <= 2.4721831748734475` | 4,808 | 19,222 | 3.997920...（图示 4.0x） |

两个箭头都是 **IVM 与 ODE-NEB 的单 metric retrospective marginal first
passage**。它们不是同一个 checkpoint 的 joint stopping test。

尤其是 panel b：OBSM-BFGS 在 `N_F=1,932` 已观察到
`Delta E=2.4623487599927207`，比 IVM 更早进入该“更低实际 barrier”条件，但它
没有达到 panel-a 的 residual target。因此 4.0x 不能表述成“IVM 相对所有六方法
最快 4.0x”；它只表示图中明确标出的 ODE/IVM pairwise 比值。

## 7. 数值输入文件

### 7.1 共同输入与 IVM 四路拼接

| 作用 | 路径 | SHA-256 |
| --- | --- | --- |
| common initial | `python/alanine_ivm_clean_four_level_20260801/inputs/common_initial_path_equidistant.npz` | `4a08066e1f0a89ad1d0ee6ecc442d7255d511217576d1ff65e9cb5e2164d891b` |
| IVM raw prefix | `python/alanine_ivm_clean_four_level_20260801/runs/stage4_inexact_optimized_resample/stage4_uniform_v3_a20_phaseA34_phaseB220.npz` | `f5be61d5121a46d349247d036ba2ffe9d258efe5caf864ee659ef2bb18aa9622` |
| IVM Pareto accepted trace | `python/alanine_ivm_clean_four_level_20260801/runs/pareto_archive/pareto_clean_stage4_v4_a20_phaseA34_phaseB220.npz` | `09ccbf7db731c96c0beda9d5879c9dd46cae2b428db8b71820aad1c56cb574fc` |
| IVM dense20 preactivation prefix | `python/alanine_ivm_clean_four_level_20260801/runs/diagnostics/dense_barrier_clean_stage4_v4_a20_phaseA34_preactivation.npz` | `398f2832bb95a5bf30168d5181dd89b03913a63360d5602d9e20b84d28d7ae74` |
| IVM dense20 accepted candidates | `python/alanine_ivm_clean_four_level_20260801/runs/diagnostics/dense_barrier_clean_stage4_pareto_screen_a20_phaseA34_phaseB220.npz` | `2e192edbb3a076dbe6311a8a81f680c9aae4e9e24aae4b8c552db58fbb54d860` |

Panel a 的 IVM early prefix 来自 combined raw，accepted tail 来自 Pareto archive；
panel b 的 early prefix 来自 dense preactivation，accepted tail 来自 dense screen。
这个拼接在 `plot_stage4_v4_pareto_all_methods.py:60-105` 定义。

### 7.2 Baselines

| 方法/作用 | 路径 | SHA-256 |
| --- | --- | --- |
| ODE path/count raw | `python/alanine_v10_chignolin_v9_fixed_alpha_20260730/formal_figure_inputs/ala_0to1_N22_k50_alpha0.0001_ode_results.npz` | `935561709ecbdfe87af31b78b45335a66545356b3e62f331a4a9fbb8125d9a09` |
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

ODE 的 `I_perp` 与 `count_seq` 来自独立 metrics NPZ，而不是 path raw NPZ；
`plot_ala_integer20_all_methods.py:174-216` 定义这一读取与 `20*count+2` 映射。

## 8. 方法配置与图中数据 horizon

| 方法 | 冻结配置 | panel-a horizon | panel-b dense horizon |
| --- | --- | ---: | ---: |
| IVM | fixed `alpha=20`; Stage-IV A34/B220; reference-free Pareto | 4,808 | 4,808 |
| ODE-NEB | ASE ODE NEB；central tangent；`k=50` | 100,022 | 100,022 |
| Static NEB | ASE static NEB；central tangent；`k=30`; `dt=0.013` | 90,002 | 60,002 |
| Original String | `dt=0.014`；每步 Appendix-A cubic reparameterization | 90,002 | 60,002 |
| OBSM-FIRE | `lambda=0.85`; inner 3; `dt=0.1`; reparam interval 1 | 90,182 | 40,502 |
| OBSM-BFGS | `lambda=0.3`; inner 1; bonded `H0`; cross-outer carry | 107,268 | 29,463 |

record selector 扫描完整 `force/iperp` 与 `barrier_force/barrier` histories，PDF
只连接其中的 strict record vertices。loader 中的 `selected_call` 主要决定
selected/path 字段，并不截断这两个 metric histories（BFGS path 的特殊截断也与
本 1x2 图无关）。

## 9. 实验生成代码

### IVM

- 核心：`paper/ActiveExperiments/alanine_66d/code/ivm/run_ivm_v9_spring_projected_production_20260730.py`
- two-phase wrapper：`paper/ActiveExperiments/alanine_66d/code/ivm/run_stage4_uniform_optimized.py`
- Pareto archive：`paper/ActiveExperiments/alanine_66d/code/ivm/build_pareto_archive.py`
- acceptance audit：`paper/ActiveExperiments/alanine_66d/code/ivm/audit_stage4_v4_acceptance.py`
- dense barrier：`paper/ActiveExperiments/alanine_66d/code/ivm/postprocess_dense_barrier.py`

冻结审阅快照的 SHA-256 分别记录在
`paper/ActiveExperiments/alanine_66d/SHA256SUMS`。实际生成数据的 Phase A/B
命令保存在：

- `python/alanine_ivm_clean_four_level_20260801/logs/stage4_uniform_v3_a20_phaseA34_phaseB220/phaseA_command.json`
  (`5ce1331b581ccbc1758e3e2623536dfa3fcc0d1ef25f75e5a35d5add934123c8`)
- `python/alanine_ivm_clean_four_level_20260801/logs/stage4_uniform_v3_a20_phaseA34_phaseB220/phaseB_command.json`
  (`6770832be5990801b9d1802f47045e28351c8a011c947c33cc6eb4dd5d013f2b`)

最终 archive 的机器 audit：
`python/alanine_ivm_clean_four_level_20260801/runs/pareto_archive/pareto_clean_stage4_v4_a20_phaseA34_phaseB220.audit.json`，
SHA-256 `4976a3253af097d64b8a257e5a04a1c22bcd33d9ec4705c893b9305df3ab6532`。

### Baselines

- Original String：`paper/ActiveExperiments/alanine_66d/code/baselines/run_string_family.py`
- Static/ODE NEB：`paper/ActiveExperiments/alanine_66d/code/baselines/run_neb_competitor.py`
- OBSM FIRE/BFGS：`paper/ActiveExperiments/alanine_66d/code/baselines/run_obsm.py`
- BFGS preconditioner：
  `paper/ActiveExperiments/alanine_66d/code/baselines/precompute_bfgs_bonded_h0.py`
  与
  `paper/ActiveExperiments/alanine_66d/code/baselines/forcefield_preconditioner.py`
- 调参与选择：
  `paper/ActiveExperiments/alanine_66d/code/baselines/run_competitor_screen.py`
  与
  `paper/ActiveExperiments/alanine_66d/code/baselines/analyze_competitor_screen.py`
- 共用物理实现：
  `paper/ActiveExperiments/alanine_66d/code/baselines/ala_core.py`

这些路径是 2026-08-04 的论文侧冻结审阅快照。可执行研究谱系位于
`python/alanine_v10_chignolin_v9_fixed_alpha_20260730/src/`。原始 NPZ 和 same-stem
audit 的哈希是结果身份的最高优先级证据；不要用今天 working source 的时间戳
倒推 2026-08-02 冻结数据。

## 10. Densified barrier

定义在 legacy
`python/alanine_v10_chignolin_v9_fixed_alpha_20260730/src/ala_core.py:698-722`：
对相邻 Cartesian images 线性加密，
逐点重新计算能量，取 `max(E)-E(start)`。postprocessor 先比较 10/20 subdivisions，
差值大于 0.05 kcal/mol 才继续到 40/80；本图所用 baseline dense audit 最终均为
20 subdivisions。

IVM 的 reference-free 接受 screen 使用 10 subdivisions 并计费；图中 accepted
barrier 使用独立保存的 `barrier_20_kcal`。因此必须区分“screen 调用计入算法成本”
与“离线 20-subdivision diagnostic 不重复计入横轴”。

## 11. 已知限制与禁止表述

### 冻结 ODE 是 mixed-generation input

当前 common initial 数组 SHA-256 是
`cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e`；
冻结 ODE `images_seq[0]` 数组 SHA-256 是
`5cc859af6b0341b58b3f24d8688d582c80cde2050da291db47ffa7aa6ae49ce2`；
最大坐标差为 `0.06150258892739646 Angstrom`。

因此这张图可称为“对冻结正式输入的六方法比较”，但**不能**称为“六方法严格从
同一个 IDPP 数组开始的 exact same-input comparison”。该 1x2 图不画路径，
这个限制不改变 CSV 中两指标的来源，但必须在科学表述中披露。机器 manifest 的
`initial_path_audit` 保存了同一结论。

### 其他边界

- ODE 有语义等价的 runner/config，但当前 bundle 没有保存产生历史 NPZ 的原始命令。
- 部分 long-budget baseline audit 的 orchestration source snapshot 字段为空；可保证
  artifact hash + runner/config 可追溯，不应承诺所有 baseline byte-identical replay。
- manifest 中的 legacy PDF、official PDF 是存在性/视觉 provenance anchor；
  Ramachandran PES cache 是 1x3 path panel 依赖。它们不是本 1x2 panel a/b 的数值输入。
- 本候选没有修改 dynamics、tuning、raw NPZ、dense NPZ 或 `paper/main.tex`。

## 12. 重绘与只刷新 provenance

完整重绘：

```bash
cd /Users/wuzhiyou/IVM_for_MEP
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/ala_cost_precision_mpl \
XDG_CACHE_HOME=/private/tmp/ala_cost_precision_cache \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_cost_to_precision_muller_style.py \
--layout 1x2 --scope all_methods
```

只刷新 CSV/manifest 而不改写已有 PDF：

```bash
cd /Users/wuzhiyou/IVM_for_MEP
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/ala_cost_precision_mpl \
XDG_CACHE_HOME=/private/tmp/ala_cost_precision_cache \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_cost_to_precision_muller_style.py \
--layout 1x2 --scope all_methods --provenance-only
```

重绘后先核对 same-stem manifest 内的 `output_pdf.sha256`、
`source_csv.sha256`、`inputs`、`plotting_code` 和 `initial_path_audit`，再考虑提升到
official。提升需要单独明确决定，并同步更新 `paper/main.tex`、正式图 manifest 与图注。
