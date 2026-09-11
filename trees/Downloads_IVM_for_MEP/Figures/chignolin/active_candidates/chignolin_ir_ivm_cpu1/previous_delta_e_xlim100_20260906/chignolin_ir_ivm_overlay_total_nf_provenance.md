# Chignolin 三指标 cost-to-precision 图溯源

当前登记日期：2026-09-06（Asia/Shanghai）。当前同 stem 图由原两联图扩为三联图：

1. (I_\perp)；
2. 沿弧长均匀加密到 500 点的能垒 (\Delta E)；
3. 原始 22 点路径上的离散 geometric action (\widehat S_h)。

三块坐标绘图区均由 Matplotlib `Axes.set_box_aspect(1)` 固定为物理宽高比 `1:1`；整页为 180 mm x 82 mm。方法、颜色和曲线定义不变，选定 IR-IVM 仍以 `IVM` 显示，历史 V9 IVM 不入图。2026-09-06 之前的两联图及其全部附件和旧溯源保存在本目录 `previous_2panel_overlay_20260906/`。

## 当前文件

| 文件 | SHA-256 |
|---|---|
| `chignolin_ir_ivm_overlay_total_nf.pdf` | `f61a5379bd96a26ae4a323a262075af78fc908e03c212c85d12719b78effe85c` |
| `chignolin_ir_ivm_overlay_total_nf.png` | `b8c045b3a631eae62c137b7f272156252c07062ff5097733f2d25990d463f61b` |
| `chignolin_ir_ivm_overlay_total_nf.svg` | `c38a24faa5eb7b6c8e855493781ae79b880320fe08c52cf152b0667604bc1d27` |
| `chignolin_ir_ivm_overlay_total_nf_frontier.csv` | `92e1b1c83d99c14a969bb1c339a569a9b3fe4b0b6a81c411db74a30c6d02b1c6` |
| `chignolin_ir_ivm_overlay_total_nf_annotations.csv` | `d714baaea66997aeeefb2b768e6ca846fc99b40518d79cec0501b1a71ce3fe6c` |
| `chignolin_ir_ivm_overlay_total_nf_provenance.json` | `f08e755fbeb553e7421054286db0f727340b4c4d3da5fd7388d3d6bf90e91a6e` |

机器可读 JSON 是当前图的完整登记；包含每种方法的源结果路径与 SHA、两类诊断文件与 audit、三个阈值、首次达到 NoF、倍率、面板顺序及 `axes_box_aspect=1.0`。CSV 保存图中实际前沿和箭头数值。

## 数据链和指标口径

实验根目录：`/Users/wuzhiyou/IVM_for_MEP/python/chignolin_ir_ivm_20260826`。

### (I_\perp)

- 来自 geometric-action 诊断中与原始路径记录逐项绑定的 `i_perp`；同一诊断已用原例子的中心差分切向量和 `local_path_weights` 复算全部 3086 条路径，最大绝对和相对差均为 0。
- 阈值：`0.21242215133173672`，为 IVM、ODE-NEB、OBSM-FIRE 三者历史最优值中最严格的共同可达值。
- 首次达到：IVM `905` NoF，ODE-NEB `17662` NoF；箭头 `19.516022099447515`，显示 `19.5x`。

### 加密后的 (\Delta E)

- 预登记：`registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json`，SHA-256 `f351020439a1ef7d75bd0c21ae6d843722d813912542bf5cef984f3f0ab424b8`。
- 数据：`runs/diagnostics/chignolin_ivm_arc_length_500.npz`，SHA-256 `3def8ffb3c70342d61285813511203c2970e30cb0146f37aa1d99aa3ce20a564`。
- audit：同 stem `.audit.json`，SHA-256 `03bca53e4390fd2a0373901ef7cea3897ffb4ff69717418924306defd17cd509`。
- 每条完整路径按未加权 (3N) 笛卡尔弧长作分段线性插值，含端点均匀取 500 点；定义为 `max(E)-E(start)`。1182 条选中路径各做 500 次全新 E/F 诊断，共 591000 次，全部排除在优化 NoF 轴之外。
- 阈值：`24.17608142601358 kcal/mol`；首次达到 IVM `905` NoF、ODE-NEB `15842` NoF；显示 `17.5x`。

### 离散 geometric action

定义为

\[
\widehat S_h=\sum_{i=0}^{20}\frac{\|\nabla V(x_i)\|+\|\nabla V(x_{i+1})\|}{2}
\|x_{i+1}-x_i\|.
\]

- 预登记：`registrations/GEOMETRIC_ACTION_DISCRETE_V1.json`，SHA-256 `87777a9461eba26693ce01903eb6993f7bf81603dddb7e771c931a029d585760`。
- 数据：`runs/diagnostics/chignolin_geometric_action_discrete_v1.npz`，SHA-256 `1c0acd1d423be9bddd77e3a4cade5008f4cff438e747d5bd83b3849e581e5921`。
- audit：同 stem `.audit.json`，记录 3086 条路径和 67892 次事后诊断 E/F 调用；全部排除在优化 NoF 轴之外。
- 使用原始 22 点、全 498 维梯度的 Euclidean norm 和分段弧长梯形求积，不用法向力代替完整梯度，也不做加密。
- 阈值严格等于 ODE-NEB 保存历史中的最小值 `101.69006490845834 kcal/mol`，不舍入。首次达到 IVM `905` NoF、ODE-NEB `20002` NoF；箭头 `22.101657458563537`，显示 `22.1x`。

该指标的完整物理计算说明和早期独立图见 [geometric-action 溯源](chignolin_ivm_geometric_action_provenance.md)。

## NoF 与跨诊断一致性

纵轴始终是原优化结果 `counter_vectors.sum(axis=1)`，包含 22 个图像的累计 force calls。弧长 500 点和 action 的所有离线诊断调用均单独审计，不加入纵轴。

合并绘图时，对弧长 500 点诊断中的每一条记录都逐项匹配 action 诊断中的：方法名、源记录索引、累计 NoF、`i_perp`、路径 SHA、源文件路径和源文件 SHA；全部一致才允许出图。历史基线源 NPZ 的原路径现已不在本地，当前重画不伪造或重新定位它们，而是验证两个不可变诊断文件及其事前登记与 audit。源文件身份仍由两个登记中的路径和 SHA 固定。

## 代码与复验

当前三联图绘图代码：

`/Users/wuzhiyou/IVM_for_MEP/python/chignolin_ir_ivm_20260826/src/plot_chignolin_ivm_three_metrics.py`

SHA-256：`feca750f2e5ceb3a9a55b0ad5c551fd692f3f417f2560c7651f9bece5a21a52e`。

它复用 `src/plot_chignolin_ivm_arc500.py` 的前沿、首次达到、样式和箭头函数；action 的物理计算代码仍为 `src/postprocess_geometric_action.py`。复验命令：

```bash
cd /Users/wuzhiyou/IVM_for_MEP/python/chignolin_ir_ivm_20260826
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  src/plot_chignolin_ivm_three_metrics.py --dry-run
```

要重画并替换同名候选图，必须先备份旧版，再显式传 `--replace`。本轮未改 `main.tex`、正式 `main.pdf`、原优化结果、两个诊断 NPZ 或其登记。
