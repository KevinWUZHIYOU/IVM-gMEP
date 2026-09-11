# Alanine 66D IVM geometric-action 1x3 图溯源

首次登记并进入 `paper0819/main.tex`：2026-09-06（Asia/Shanghai）；同日将
trajectory marker 冻结为论文 Fig. 4 的共享方法样式。

## 1. 登记对象与真值优先级

本文件登记以下图及其同 stem 机器附件：

`/Users/wuzhiyou/IVM_for_MEP/output/pdf/alanine66d_ivm_geometric_action_1x3.pdf`

状态：**已验证，并以冻结副本进入
`/Users/wuzhiyou/Downloads/IVM_for_MEP/paper0819/main.tex`，标签为
`fig:Ala2_convergence`。** 它没有替换源仓库 `/Users/wuzhiyou/IVM_for_MEP/paper/main.tex`
中的正式 Ala 图。

若人工文字与机器附件冲突，按以下顺序判断：

1. same-stem `_manifest.json` 中的输入路径、控制参数和 SHA-256；
2. same-stem `_source.csv` 中每个绘图顶点的来源文件、记录序号和成本；
3. geometric-action v5 preregistration、结果 NPZ 及 audit；
4. 本 Markdown 的解释性文字。

## 2. 图、附件、代码与哈希

以下哈希均在最终画图脚本把状态改为
`registered_and_cited_in_paper0819` 并重生成三件套后计算。

| 角色 | 绝对路径 | SHA-256 |
| --- | --- | --- |
| 登记 PDF | `/Users/wuzhiyou/IVM_for_MEP/output/pdf/alanine66d_ivm_geometric_action_1x3.pdf` | `7c9f7c82c43eba9e7b86245c4a584dbca80caeedb6efcc598ae605fc18b223ae` |
| 逐顶点数据 | `/Users/wuzhiyou/IVM_for_MEP/output/pdf/alanine66d_ivm_geometric_action_1x3_source.csv` | `10fd5aab939d0d45ae4d6ca90ebce189d4bb1aed6de21bfe4cbdf2ef323eeca6` |
| 机器 manifest | `/Users/wuzhiyou/IVM_for_MEP/output/pdf/alanine66d_ivm_geometric_action_1x3_manifest.json` | `61f9e08bf0ac1550b073082992333383aea53f3798e46ae6e20c4c6cd6fac7be` |
| 1x3 plotter | `/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_delta_aware_geometric_action_1x3.py` | `55497f60d3a95ee74525d486b15bada9b3c8d5ccaf9feb5816d9d7f85ac88678` |
| action evaluator | `/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/alanine_66d/code/metrics/postprocess_ala_geometric_action.py` | `ee21e6f3ae143fbf8bb58277e7cd828c4259d2f0940085feda447f2f9ebad430` |
| evaluator unit test | `/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/alanine_66d/code/metrics/test_postprocess_ala_geometric_action.py` | `3de86f62963f4aa6ded5e046b7cefee77be93d024541f4419bc13859d2df7b32` |
| v5 preregistration | `/Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801/reports/ala_discrete_geometric_action_v5_preregistered.json` | `48eb99096e6aa1e569f6ef15a3b5c86dac0affcbaea2c563df42f60dcd55ffba` |
| action result | `/Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801/runs/diagnostics/ala_discrete_geometric_action_v5.npz` | `d19d8efacf4b48b31ace4a87ff6857b04f65c6be9b3a1172db9e4506f58076eb` |
| result audit | `/Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801/runs/diagnostics/ala_discrete_geometric_action_v5.audit.json` | `797e24ffc28bd8684fccdb8b689089057dd6dd188ac53d80b383d387343c5ee3` |

`_source.csv` 共 18,056 个数据行：panel a 为 13,914 行，panel b 为 118 行，
panel c 为 4,024 行。每一行登记 method、实际 metric、running-best metric、最小
`Number of Forces`、源记录序号、路径 SHA-256、成本来源和 metric 来源。

论文目录保留同名 PDF、CSV、manifest 和本溯源文件的冻结副本：

`/Users/wuzhiyou/Downloads/IVM_for_MEP/paper0819/Figures/ala2_66d/`

冻结 PDF 与登记 PDF 的 SHA-256 相同；CSV 和 manifest 亦逐字节相同。

## 3. 三个 panel 的数据链

```text
alanine66d_ivm_geometric_action_1x3.pdf
├── panel a: discrete I_perp
│   └── alanine66d_delta_aware_grid_ivm_replacement_1x2_preview_source.csv
├── panel b: densified Delta E
│   └── 同一 1x2 source CSV / manifest 所登记的离线加密结果
└── panel c: discrete geometric action
    └── ala_discrete_geometric_action_v5.npz
        ├── v5 preregistration
        ├── v5 audit
        └── 5,388 条已登记的 22-image 路径及其优化成本来源
```

两个旧 panel 的冻结入口为：

- `/Users/wuzhiyou/IVM_for_MEP/output/pdf/alanine66d_delta_aware_grid_ivm_replacement_1x2_preview_source.csv`
  (`93e9a938c289f964fb509f57ab861c8ce4e93e612c0f22820b2b5a699643176d`)；
- 同 stem `_manifest.json`
  (`b6b8ebd220d81e750c0b5d9c6f4953df87fc9be00d30819ccac740c15e82c6b7`)。

manifest 继续登记被选 IVM 结果、20 个 grid-metric task artifacts 以及五条冻结 baseline
的准确路径与 SHA-256；无需依赖本文件手工抄写超长输入清单。

### 与论文 Fig. 4 一致的 trajectory marker

Fig. 6 的 shared-method marker 直接冻结为 Fig. 4 的映射和尺寸；白色 marker face、
`markeredgewidth=0.7`，并以每条曲线约 8 个 marker 的策略显示：

| Fig. 6 方法 | marker | size |
| --- | --- | ---: |
| IVM | circle `o` | 3.1 |
| ODE-NEB | square `s` | 2.9 |
| Static NEB | down triangle `v` | 2.8 |
| Original String | diamond `D` | 2.7 |
| OBSM-FIRE | square `s` | 2.8 |
| OBSM-BFGS | filled-plus symbol `P` | 2.8 |

机器 manifest 同时登记 Fig. 4 实际 plotter 和其共享方法 style source 的路径及哈希；
plotter 会在方法集合或映射不全时 fail closed。

## 4. 被选 IVM 与作用量定义

蓝线图例简写为 `IVM`。实际算法是 inexact + reparameterization 的离散 IVM，
从共同 IDPP 初始路径出发，参数为

```text
alpha = 10
r = 0.7
m_max = 5
total internal-image force budget = 5000
```

体系为真空 Amber14/OpenMM alanine dipeptide：22 个原子、66D；路径含 20 个可动内部
images 和两个固定端点，共 22 个 images。对原始多边形路径
`x_0,...,x_21`，令 `F_i=-grad V(x_i)`，图中离散 geometric action 为

\[
\widehat S_h
=\sum_{i=0}^{20}\frac{\lVert F_i\rVert+\lVert F_{i+1}\rVert}{2}
  \lVert x_{i+1}-x_i\rVert .
\]

- 积分采用 polygonal arc-length trapezoid；端点以相邻半段权重计入；
- 使用完整物理梯度范数，不做 normal projection；
- 单位由 eV 转成 kcal/mol；
- action 只用原始 22 个节点，**不加密**；
- tangent 与论文示例和生产指标一致，为 normalized central difference
  `x[i+1]-x[i-1]`。tangent 用于核对路径几何与 `I_perp`；action 本身依赖完整梯度范数。

## 5. 离线计算、审计和 NoF 口径

evaluator 对 5,388 条已登记路径逐条调用同一个 OpenMM CPU context，每条恰好评价
22 个 images，总计

```text
5,388 * 22 = 118,536 offline diagnostic force evaluations.
```

这些调用只计算图中诊断量，不改变优化状态，也不计入纵轴 `Number of Forces`。
计算固定 `jobs=1`、`OPENMM_CPU_THREADS=1`，并持有 host-local Ala `fcntl` 锁。
Audit 验证：

- 非 ODE 历史的重算 `I_perp` 最大绝对差为
  `1.2578104247040756e-07`，最大相对差为 `2.634866361558543e-07`；
- ODE-NEB 的 legacy `loss_perp` 早于统一 tangent 定义，因此不拿它做同一 `I_perp`
  数值复核；记录 1--4637 的保存力则已与当前 OpenMM 复核，最大相对 L2 差
  `4.265928532046815e-05`、最大绝对差 `3.981687775267417e-06 eV/Angstrom`；
- ODE 记录 0 的 `F_original[0]` 是全零占位，不冒充已验证的历史力；该路径的 action
  使用本次 OpenMM 重算力。

### NoF 口径更正

v5 preregistration 的 `optimization_cost` 说明把所有来源概括为“all 22 images including
endpoints”，这个句子对被选 IVM 过宽。冻结 contract 不回写；准确口径由运行结果和
same-stem CSV 保留：

- IVM 纵轴使用 `total_internal_joint_evaluations`，统计 20 个可动内部 images，不含
  两个固定端点的一次性初始评价；
- ODE-NEB 使用 `20*count+2`，其余历史 baseline 使用冻结的 all-image counter ledger，
  均含两个端点各一次；
- 因而 IVM 与历史 baseline 存在固定 2 次端点评价的口径差。它不改变图中一位小数的
  `24.7x`、`28.6x`、`40.9x` 显示，但不得把纵轴解释成端点口径完全统一。

## 6. 图上 first-passage 读数

曲线只连接实际保存的 strict running-lowest records，不插值。三个阈值分别回答不同
metric 的 first-passage 问题，不要求来自同一个路径 checkpoint。

| panel | 阈值 | IVM | ODE-NEB | ODE-NEB / IVM |
| --- | ---: | ---: | ---: | ---: |
| a, `I_perp` | `2.1e-4` | 3,386 | 83,522 | `24.7x` |
| b, densified `Delta E` | `2.4721831748734475 kcal/mol` | 672 | 19,222 | `28.6x` |
| c, `S_hat_h` | `5.0 kcal/mol` | 2,233 | 91,382 | `40.9x` |

panel c 的 IVM 最低记录为 `4.906956048883713 kcal/mol`，ODE-NEB 为
`4.942399719945519 kcal/mol`。`5.0 kcal/mol` 是计算完成后为便于画图选取的圆整展示
阈值，不是预注册的模型选择阈值。

## 7. 验证与重建

单元测试：

```bash
cd /Users/wuzhiyou/IVM_for_MEP
PYTHONDONTWRITEBYTECODE=1 \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python -m unittest \
paper/ActiveExperiments/alanine_66d/code/metrics/test_postprocess_ala_geometric_action.py
```

历史离线诊断采用先登记、后计算的 fail-closed 两步流程：

```bash
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
paper/ActiveExperiments/alanine_66d/code/metrics/postprocess_ala_geometric_action.py --register
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
paper/ActiveExperiments/alanine_66d/code/metrics/postprocess_ala_geometric_action.py --compute
```

现有 v5 contract/result/audit 是冻结证据，脚本会拒绝原地覆盖；真正重算必须使用新的
版本化输出名并重新登记，不能删除 v5 后静默重跑。

从已验证 v5 结果重画登记图：

```bash
cd /Users/wuzhiyou/IVM_for_MEP
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/ala_geometric_action_mpl \
XDG_CACHE_HOME=/private/tmp/ala_geometric_action_cache \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_delta_aware_geometric_action_1x3.py \
--overwrite
```

论文预览按目标目录的 `AGENTS.md` 构建到隔离目录，不覆盖正式 `main.pdf`：

```bash
cd /Users/wuzhiyou/Downloads/IVM_for_MEP/paper0819
/Library/TeX/texbin/latexmk -pdf -synctex=1 -interaction=nonstopmode \
-file-line-error -halt-on-error -outdir=tmp/vscode-latex main.tex
```
