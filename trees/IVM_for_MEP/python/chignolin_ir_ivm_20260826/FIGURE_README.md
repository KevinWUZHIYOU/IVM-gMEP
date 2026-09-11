# Chignolin IVM replacement figure

本文件登记当前候选图
`Figures/chignolin/active_candidates/chignolin_ir_ivm_cpu1/chignolin_ir_ivm_overlay_total_nf.pdf`
的生成链。图中标为 `IVM` 的曲线来自正式 IR-IVM 单线程结果；旧 V9 IVM 已删除。

该图的自包含单图登记位于
`/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/chignolin/active_candidates/chignolin_ir_ivm_cpu1/chignolin_ir_ivm_overlay_total_nf_provenance.md`。
实验根 `figures/active_candidates/` 中 SHA-256 为 `0be0fb35...db473` 的同名旧 PDF 属于历史
6 方法/adaptive-barrier 版本，不是当前登记图。

## 固定输入与指标协议

- 比较合同：`registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json`。
- 原始优化结果：合同中的五个 `source_result` 及其 SHA256；IVM 使用
  `runs/repro_cpu1/ivm_chig_irivm_a1e1_r05_m05_inf5000_cpu1_prefix_central_state_ivm_cfg844e61a3b6af_budget5000.npz`。
- 势垒指标：`runs/diagnostics/chignolin_ivm_arc_length_500.npz` 及同名
  `.audit.json`。每条入图路径都在未加权 \(3N\) 笛卡尔空间沿累计弧长均匀取
  500 个总点（含两个端点），逐点执行新的 OpenMM CPU/double 联合计算。
- 诊断覆盖：1182 条路径，每条恰好 500 次，合计 591000 次；这些诊断调用不计入优化横轴。
- 文章成本：每个状态的 `counter_vectors.sum(axis=1)`，包含全部 22 个图像的实际调用。
- 可报告 IVM 状态：初始状态或 `reparam_committed=True` 的完整重参数化状态。

共同精度阈值及首次达到 Number of Forces：

| 面板 | 阈值 | IVM | ODE-NEB | OBSM-FIRE |
|---|---:|---:|---:|---:|
| \(I_\perp\) | 0.21242215133173672 | 905 | 17662 | 19942 |
| \(\Delta E_{500}\) | 24.17608142601358 kcal/mol | 905 | 15842 | 19942 |

图中倍率箭头只比较 IVM 与 ODE-NEB，因此分别标为 \(19.5\times\) 和
\(17.5\times\)。OBSM-FIRE 保留为曲线与共同阈值证据，不画倍率箭头。

## 样式与生成代码

- 作图脚本：`src/plot_chignolin_ivm_arc500.py`。
- 样式参照：
  `/Users/wuzhiyou/IVM_for_MEP/output/pdf/alanine66d_delta_aware_grid_ivm_replacement_1x2_preview.pdf`
  （SHA256 `483530e8c0d764e2c4b4c82af3c280cc7f8fee300eb35b12db6e1260759f3a44`）。
- 保持参照图的 180 mm × 72 mm 画布、字号、开口标记、线型、三列两行图例、
  `k` 格式纵轴、竖直双向箭头和箭头中部右侧倍率标签。
- 输出目录：
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/chignolin/active_candidates/chignolin_ir_ivm_cpu1/`。

先只读验证，再生成：

```bash
CHIG_IR_ROOT=/Users/wuzhiyou/IVM_for_MEP/python/chignolin_ir_ivm_20260826
CHIG_IR_PY=/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python
FIGDIR=/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/chignolin/active_candidates/chignolin_ir_ivm_cpu1
cd "$CHIG_IR_ROOT"

"$CHIG_IR_PY" src/plot_chignolin_ivm_arc500.py \
  --contract registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json \
  --arc500 runs/diagnostics/chignolin_ivm_arc_length_500.npz \
  --output-dir "$FIGDIR" --stem chignolin_ir_ivm_overlay_total_nf --dry-run

"$CHIG_IR_PY" src/plot_chignolin_ivm_arc500.py \
  --contract registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json \
  --arc500 runs/diagnostics/chignolin_ivm_arc_length_500.npz \
  --output-dir "$FIGDIR" --stem chignolin_ir_ivm_overlay_total_nf
```

生成时同时写出 PDF、PNG、SVG、前沿 CSV、箭头 CSV 和 provenance JSON。
精确文件哈希及全部原始输入映射见 `RESULTS_AND_PROVENANCE.md`。该脚本只写候选图目录，
不会覆盖正式文章图或修改 `main.tex`。
