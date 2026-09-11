# Chignolin 离散 geometric action：计算与图溯源

登记日期：2026-09-04（Asia/Hong_Kong）。本次是既有实验的离线指标计算，不重新运行优化算法，也不替换原来的 I_perp/Delta E 图。

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

共同作用量阈值为 `108.65188139033872 kcal/mol`。首次达到和各条保存轨迹中的最优值如下；不同方法的最终预算并不相同，末列的最优值不能解释成同预算比较。

| 方法 | 首次达到共同作用量阈值 NoF | 最后可报告状态 NoF | 最优离散作用量 (kcal/mol) |
|---|---:|---:|---:|
| IVM | 905 | 3740 | 53.917860845503846 |
| ODE-NEB | 17662 | 20002 | 101.69006490845834 |
| Static NEB | 未达到 | 20002 | 189.90779248422842 |
| Original String | 未达到 | 20002 | 190.20069796028702 |
| OBSM-FIRE | 19942 | 19982 | 108.65188139033872 |

IVM 在前一完整状态 NoF 735 的作用量为 `130.09014975478632`，在 NoF 905 降到 `99.37761760173404`。
ODE-NEB 在 NoF 17622 的值为 `114.1585791275863`，在 NoF 17662 为 `108.37902458694725`。
因此右图箭头为 `17662 / 905 = 19.516022099447515`，显示 `19.5x`；左图原有 I_perp 阈值的倍率也是 `19.5x`，这两项相同是实际数据结果，并非复用旧图右面板的 `17.5x`。

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
| 新绘图代码 | `src/plot_chignolin_geometric_action.py` | `5b9456e5e75f3da6145d4f7422ff9d8397b2e66ebaefebc79861e79e45fa424c` |

诊断数据位于 `runs/diagnostics/chignolin_geometric_action_discrete_v1.npz`、同 stem `.csv` 和 `.audit.json`。CSV 每行对应一个原始完整路径；NPZ 保存逐节点范数和逐段长度，能直接重建积分，不需要再次请求力。

新图位于本目录 `chignolin_ivm_geometric_action_1x2.pdf`；PNG、SVG、`_frontier.csv`、`_annotations.csv` 和 `_provenance.json` 同步保存。每个前沿点还保存源 NPZ 路径、源记录索引及路径 SHA。

| 数据/图附件 | SHA-256 |
|---|---|
| 诊断 NPZ | `1c0acd1d423be9bddd77e3a4cade5008f4cff438e747d5bd83b3849e581e5921` |
| 诊断 CSV | `06c21d74969c27a7e7899fe366e10eec9da965223794d16824628c1e9d80e142` |
| 诊断 audit JSON | `8398b087bb79897839154024fda08cba82eb417d5143b354a78906ab2f7c9d14` |
| 新图 PDF | `3a1abffd9f0f805e0c0a60cf8c983857833aea717b35f0977b218cf2e4d66c4d` |
| 新图 PNG | `100af921d549717c93560628ade1e2fca779cc4e6a5f7733c0e2ea33d16a239c` |
| 新图 SVG | `aaf1974cd4e04c389a39c4ea717e8576b87f7875369e1ebe1c05a9000da694b3` |
| 新图前沿 CSV | `b03229a66061e29b28887e81bf766bb72c1dec5b5f39fcac209f58f277468cb9` |
| 新图箭头 CSV | `7012c97a1749b77edb70b45a22c6efecaefad8e53101a24214af1a35319a4504` |
| 新图 provenance JSON | `834beacd0074b9111d8bb56bf96bd73473f2e686f9e4b5c2a0048049df6c43f4` |

右面板使用逆向对数横轴（100000 到约 32.35），以显示完整的离散作用量下降；其他版式、图例、颜色与箭头位置沿用原图。PDF 已完成逐页渲染目视检查；64 项单元测试通过。

共同阈值的规则沿用原图：对 IVM、ODE-NEB、OBSM-FIRE，取各自历史最优值中的最大者；箭头只比较 ODE-NEB/IVM 的首次达到 NoF。不是挑选对 IVM 最有利的阈值，也不把线段插值当成实测首次达到。

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

## 解释边界

这是用户要求的 22 点离散近似，未做网格收敛检查，不应等同于连续积分的精确值。它与 500 点势垒是两个不同指标；不能用该值代替 `Delta E` 或当作离散优化已收敛的证明。仅新增指标和图，不修改 `main.tex`、原图、原始优化结果或已有 500 点数据。
