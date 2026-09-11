# Chignolin 离散 geometric action：计算与图溯源

登记日期：2026-09-04（Asia/Hong_Kong）。本次是既有实验的离线指标计算，不重新运行优化算法，也不替换原来的 I_perp/Delta E 图。

2026-09-04 最新更新：按用户明确要求，右图阈值严格等于 ODE-NEB 所有保存状态的最小离散作用量，即 `S_h <= 101.69006490845834 kcal/mol`，不取整、不再使用 102。左图、路径数据和诊断计算均不变。上一版阈值 102 的图、附件、本文及绘图代码保存在本目录 `previous_action_threshold102_20260904/`；更早的自动阈值版本保存在 `previous_action_threshold108p651881_20260904/`。此次只改绘图阈值，不修改原诊断预登记；下文记录当前图的口径。

## 指标定义

按用户指定的公式，计算完整物理梯度的范数积分：

\[
\widehat S(\varphi)=\int_\varphi |\nabla V|\,d\ell,
\qquad
\widehat S_h=\sum_{i=0}^{20}\frac{\|\nabla V(x_i)\|+\|\nabla V(x_{i+1})\|}{2}\,
\|x_{i+1}-x_i\|.
\]

- 使用原始 22 个图像，498 维未加权笛卡尔空间；不插入加密点、不拟合样条、不改变路径。
- 用全 498 维梯度的 Euclidean norm，不是逐原子范数之和，也不是法向投影的范数。
- 每段做弧长梯形求积，包括两个端点的半段权重。内点权重严格等于原代码 `local_path_weights(path)`。
- `F=-grad V`，所以 `norm(F)=norm(grad V)`；内部单位是 eV/Angstrom 乘 Angstrom，按原代码 `EV_TO_KCAL=23.0605` 换成 kcal/mol。
- 只计算上述定义；不添加其他 geometric-action 定义里可能出现的 `grad V dot tangent` 项。

切向量直接调用原例子的 `chignolin_core.central_tangents`：

\[
\tau_i=\frac{x_{i+1}-x_{i-1}}{\|x_{i+1}-x_{i-1}\|}.
\]

本指标本身不要求把梯度投影到切向或法向。切向量用于逐路径复算原 `I_perp` 并对照原结果，以检验路径、力和代码口径一致；它不改变 `widehat S` 中的完整梯度。新图左面板仍用原始已保存的 `I_perp`，右面板用新计算的 `widehat S_h`。

## 原始实验与取样

论文工作树：`/Users/wuzhiyou/Downloads/IVM_for_MEP`。

实验根目录：`/Users/wuzhiyou/IVM_for_MEP/python/chignolin_ir_ivm_20260826`。下文 `src/`、`runs/`、`registrations/` 相对路径均以此为基准。

五个原始优化结果与旧图完全相同；其路径和 SHA-256 由以下两个文件固定：

- `registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json`：只复用其原始实验身份，不复用 500 点取样或势垒值。
- `registrations/GEOMETRIC_ACTION_DISCRETE_V1.json`：本轮物理计算前登记的公式、源码哈希、路径选择哈希、单线程设置和预算。

具体源 NPZ、实验代码、预登记和参数细节见同目录 [原图溯源](chignolin_ir_ivm_overlay_total_nf_provenance.md)。IVM 仍是 `alpha=10, r=0.5, m_max=5, eta=1.1` 的单线程正式结果，强制重参数化；旧 V9 IVM 不入图。

| 方法 | 计算路径数 | 选择规则 |
|---|---:|---|
| IVM | 12 | 初始状态或 `reparam_committed=True` 的完整状态 |
| ODE-NEB | 906 | 全部保存状态 |
| Static NEB | 1000 | 全部保存状态 |
| Original String | 1000 | 全部保存状态 |
| OBSM-FIRE | 168 | 全部保存状态 |

本次离散计算比旧右面板的时间取样更完整：Static NEB 和 Original String 不再只取 48 个状态，而是各取 1000 个保存状态。所有方法的空间离散统一为原始 22 点。

## 计费与一致性检查

每条路径的全部 22 个点都通过原 `OpenMMBackend.evaluate_joint` 新计算一次，单线程、同一 Amber14/GBn2 模型、同一 topology。记录结果含每个点的梯度范数、能量、每段长度、原路径 SHA、源 NPZ SHA、源记录索引和 22 图像计数向量。

共 3086 条路径，恰好 67892 次新的诊断 E/F 调用；包含每条路径的两个端点，不复用诊断缓存。它们单独计费，不加入优化 `Number of Forces` 轴，也不用于优化接受/拒绝决策。

图中优化成本仍为原 NPZ 的 `counter_vectors.sum(axis=1)`。原有 rejected trial、recovery 和未闭合尾段的收费没有更改；未提交状态不会因新增指标较好而进入 IVM 曲线。

本轮使用独立 host-local `fcntl` 锁 `logs/chignolin_ir_ivm_host.lock`、`jobs=1` 和 `OPENMM_CPU_THREADS=1`。代码沿用生产 backend 的 `precision='double'` 请求；原 core 在 CPU 平台不设置 GPU 的 `Precision` 属性，因此这里只声称与生产 CPU backend 一致，不另行声称 OpenMM CPU 的每个内部计算核均为双精度。

所有路径的 `I_perp` 用相同中心差分切向量和 `local_path_weights` 复算；预登记验收容差为 `rtol=1e-5, atol=1e-8`。3086 条路径实测最大绝对误差和最大相对误差均为 0：复算值与保存值完全相同。

## 计算结果

当前作用量阈值按用户要求取 ODE-NEB 的历史最小值 `101.69006490845834 kcal/mol`，不再使用三个参考方法都能达到的自动阈值，也不采用近似的 102。首次达到和各条保存轨迹中的最优值如下；不同方法的最终预算并不相同，末列的最优值不能解释成同预算比较。

| 方法 | 首次达到 ODE-NEB 最小作用量阈值的 NoF | 最后可报告状态 NoF | 最优离散作用量 (kcal/mol) |
|---|---:|---:|---:|
| IVM | 905 | 3740 | 53.917860845503846 |
| ODE-NEB | 20002 | 20002 | 101.69006490845834 |
| Static NEB | 未达到 | 20002 | 189.90779248422842 |
| Original String | 未达到 | 20002 | 190.20069796028702 |
| OBSM-FIRE | 未达到 | 19982 | 108.65188139033872 |

IVM 在前一完整状态 NoF 735 的作用量为 `130.09014975478632`，在 NoF 905 降到 `99.37761760173404`。
ODE-NEB 在前一状态（原记录索引 904）NoF 19962 的值为 `109.02594599809431`，在最后状态（原记录索引 905）NoF 20002 达到全程最小值 `101.69006490845834`。该值直接从数组取最小值，不对阈值作舍入。
因此右图箭头为 `20002 / 905 = 22.101657458563537`，显示 `22.1x`，箭头的 ODE-NEB 端点严格落在其最小作用量对应的点。左图仍用 `I_perp <= 0.21242215133173672`，首次达到为 IVM 905、ODE-NEB 17662，保持 `19.5x`。两个箭头均由各自指标的首次达到记录独立计算。

原始 IVM 的 12 个状态：

| 原记录索引 | NoF | 离散作用量 (kcal/mol) |
|---:|---:|---:|
| 0 | 22 | 60336.99184701898 |
| 7 | 126 | 8369.997345345875 |
| 11 | 210 | 843.9523103106727 |
| 20 | 386 | 306.06864980319074 |
| 30 | 567 | 168.96766339353093 |
| 40 | 735 | 130.09014975478632 |
| 50 | 905 | 99.37761760173404 |
| 68 | 1220 | 89.45265289184005 |
| 82 | 1455 | 74.54346668961087 |
| 102 | 1697 | 68.47740765619747 |
| 112 | 1947 | 62.5509983703985 |
| 258 | 3740 | 53.917860845503846 |

## 当前代码与数据链

| 角色 | 路径 | SHA-256 |
|---|---|---|
| 事前诊断登记 | `registrations/GEOMETRIC_ACTION_DISCRETE_V1.json` | `87777a9461eba26693ce01903eb6993f7bf81603dddb7e771c931a029d585760` |
| 离散计算代码 | `src/postprocess_geometric_action.py` | `959167006d816b4346b9b8a569d2349632a256309340c7ff558511855b6dc1e0` |
| 原模型、切向量、权重 | `src/chignolin_core.py` | `ad5d6532f7acd7c7b2560fdb68e6ead796374bcf0b24d1c0790b7e51fa5e4faa` |
| 原图样式/前沿/首次达到代码 | `src/plot_chignolin_ivm_arc500.py` | `62d30b5b81e0ab3b42bdf080308e95c615a4ecee0b960946b1df4dd15e0743b9` |
| 新绘图代码 | `src/plot_chignolin_geometric_action.py` | `0946c13524bdc1d769210d183f5f33e431840d2c4a7811848a86f441f22d6988` |

诊断数据位于 `runs/diagnostics/chignolin_geometric_action_discrete_v1.npz`、同 stem `.csv` 和 `.audit.json`。CSV 每行对应一个原始完整路径；NPZ 保存逐节点范数和逐段长度，能直接重建积分，不需要再次请求力。

新图位于本目录 `chignolin_ivm_geometric_action_1x2.pdf`；PNG、SVG、`_frontier.csv`、`_annotations.csv` 和 `_provenance.json` 同步保存。每个前沿点还保存源 NPZ 路径、源记录索引及路径 SHA。

| 数据/图附件 | SHA-256 |
|---|---|
| 诊断 NPZ | `1c0acd1d423be9bddd77e3a4cade5008f4cff438e747d5bd83b3849e581e5921` |
| 诊断 CSV | `06c21d74969c27a7e7899fe366e10eec9da965223794d16824628c1e9d80e142` |
| 诊断 audit JSON | `8398b087bb79897839154024fda08cba82eb417d5143b354a78906ab2f7c9d14` |
| 新图 PDF | `813848e3a3b9436c78830a0ba3c31b669b7835ff814bd6f1b225c0717771cdf6` |
| 新图 PNG | `41f1d53056a0c8382b21ed3e88bb092f2093fb24f0805b01a69a256df32909cb` |
| 新图 SVG | `4fe06b2196480062900c877bf6dc5a0c7e9609a4319e4056c321881fa787d419` |
| 新图前沿 CSV | `b03229a66061e29b28887e81bf766bb72c1dec5b5f39fcac209f58f277468cb9` |
| 新图箭头 CSV | `2d85f8092e201d2b30e3b25dbd146eba17bb1a7aaa81206d326f2e7dee8ee8ca` |
| 新图 provenance JSON | `e40772bc4d9549a10506725d361a5ed97ac11415fd1c24ced53f12ad53b9d65d` |

右面板使用逆向对数横轴（100000 到约 32.35），以显示完整的离散作用量下降；其他版式、图例、颜色与箭头位置沿用原图。右横轴标记为 `S_h`。原计算通过 64 项单元测试；本次阈值更新另已验证阈值严格等于 ODE-NEB 数组最小值、四个首次达到 NoF、OBSM-FIRE 未达到该阈值，以及显式指定 102 仍可复现上一版 NoF，并重新渲染目视检查。前沿 CSV 哈希不变，说明仅修改阈值注释而未改变曲线数据。

左图阈值的规则沿用原图：对 IVM、ODE-NEB、OBSM-FIRE，取各自历史最优值中的最大者。右图改为 ODE-NEB 自身的历史最小作用量，覆盖原诊断登记中的自动绘图阈值规则，但不改变登记的指标定义、样本选择、原始路径或计费。生成的 JSON 用 `threshold_policy` 明确记录此覆盖。箭头只比较 ODE-NEB/IVM 的首次达到 NoF，不把线段插值当成实测首次达到，也不将单阈值倍率解释成全程加速比。

## 复验命令

```bash
cd /Users/wuzhiyou/IVM_for_MEP/python/chignolin_ir_ivm_20260826
CHIG_ACTION_PY=/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python

# 已有登记和数据时，先进行不写图、不重复计算力的验证
"$CHIG_ACTION_PY" src/plot_chignolin_geometric_action.py --dry-run

# 用不同目录重画，不覆盖现有候选图
"$CHIG_ACTION_PY" src/plot_chignolin_geometric_action.py \
  --output-dir /private/tmp/chignolin_action_replot

"$CHIG_ACTION_PY" -m unittest discover -s tests -p 'test_geometric_action.py' -v
```

原始执行顺序是 `postprocess_geometric_action.py --register`，再 `--compute`；两个入口均拒绝覆盖既有登记/诊断。要再次做物理重算，先在独立实验副本中登记新输出，而不是删除现有科学证据。

不传 `--action-threshold` 时，绘图默认阈值现在严格等于 ODE-NEB 的历史最小作用量；`--action-threshold 102` 只用于复现上一版。若需显式替换当前图，必须先保留旧版，再传 `--replace`。本次已将旧版完整备份，原始 I_perp/Delta E 图未变。

## 解释边界

这是用户要求的 22 点离散近似，未做网格收敛检查，不应等同于连续积分的精确值。它与 500 点势垒是两个不同指标；不能用该值代替 `Delta E` 或当作离散优化已收敛的证明。仅新增指标和图，不修改 `main.tex`、原图、原始优化结果或已有 500 点数据。
