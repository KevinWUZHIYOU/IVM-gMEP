# Barnase-Barstar joint-E/F cost-to-precision 图溯源草案

状态：`PREPARED_PLOTTING_LAYER_ONLY_NOT_A_FROZEN_SCIENTIFIC_FIGURE`

本文件登记最终二维 cost-to-precision 图的绘图与审计协议。当前只完成零
OpenMM 的绘图层、pack 登记器、figure contract 冻结器、合成预览及测试；
尚未把未完成的 IVM/ODE-NEB 后期 joint-E/F 数据登记为正式输入，也没有冻结
最终 figure contract。

## 图回答什么

图只比较 `IVM` 与 `ODE-NEB`：

- (a) 横轴为原始 32-image 路径上的离散 \(I_\perp\)，纵轴为
  `Number of Forces`；
- (b) 横轴为同一条 32-image 路径按未加权 Cartesian arc length 均匀取
  500 点后得到的 \(\Delta E\)，纵轴同样为 `Number of Forces`；
- 每个实际测得的点都以淡色空心 marker 保留；粗线只连接这些观测点组成的
  observed lower envelope；
- 不插值，不把两个相邻观测 NoF 之间的线段解释为精确 first passage；
- 箭头比较的是共同可达精度下的 earliest **observed** passage。可见标签只写
  `...x`，位于竖直箭头线中点的右侧；“observed-only”限定完整写入 sidecar
  provenance，而不是挤进图内；
- joint-E/F 后处理调用全部是 diagnostic，`optimization N_F` 贡献严格为 0。

## 每个绘图点的强制闭包

每个点必须经过 `plot_bb_cost_to_precision_joint_ef_dense.py` 的逐数组复核：

1. authoritative shard 是不可变、hash-bound 的 `.npz`；
2. `selected_path` 是有限的 `(32, 9396)` float64 数组；
3. `dense_coordinates_A`、`dense_energy_eV`、
   `dense_force_eV_per_A` 分别是 `(500,9396)`、`(500,)`、
   `(500,9396)` 的有限 float64 数组；
4. dense 坐标逐字节等于从该 `selected_path` 确定性重建的 Cartesian arc500；
5. 坐标、能量和完整力的数组 SHA-256 与 shard/manifest/aggregate 一致；
6. \(\Delta E=(\max E-E_0)\times23.0605\) 与 shard/aggregate 一致；
7. ODE-NEB 的离散 \(I_\perp\) 从同一 32-image 路径和 30 个内部 image 的
   已保存完整力重算；IVM 若未保存这 30 个历史力，则必须由同一 stored path
   的 exact replay result 与 PASS verification 联合闭合 path hash、history
   index、NoF、\(I_\perp\) 和 complete/non-terminal 状态；
8. contract、result、audit、manifest、WAL、evaluator、topology、`bb_core` 和
   shard 全部是直接 SHA-256 binding；
9. 每个 bound pack 的全部点进入图，禁止看到指标后从 pack 内删点；同一方法
   的重复 NoF 直接拒绝。

## 已有和待生成的输入

| 方法 | 预期 NoF | 当前状态 | plotting pack spec / 预期 bundle |
|---|---:|---|---|
| ODE-NEB early | 60, 150, 300, 1320 | 完整并已通过真实零 OpenMM pack 复核 | `registrations/diagnostics/BB_COST_PRECISION_PACK_ODE_EARLY_JOINT_EF_20260902_R1.json` |
| IVM early | 60, 150, 300, 1329 | calls 2/5 已有完整 shard；calls 10/50 的 R2 continuation 尚未正式 PASS | future `runs/diagnostics/bb_irivm_short_four_arc500_joint_ef_20260902_r2.{npz,manifest.json,audit.json}` |
| ODE-NEB late | 2040, 4080, 8160, 9990 | execution contract 已冻结，尚未产生物理结果 | future `runs/diagnostics/bb_ode_neb_late_four_arc500_joint_ef_20260902_r1.{npz,manifest.json,audit.json}` |
| IVM late | 约 6 个预登记后期点 | package/evaluator 已冻结，selection/result 尚不存在；不得当作结果 | future `runs/diagnostics/bb_irivm_late_six_arc500_joint_ef_20260902_r1.{npz,manifest.json,audit.json}` |

默认 figure freezer 要求每种方法 8--12 个完整观测点。这个门禁允许预期的
ODE-NEB 8 点和 IVM 约 8--10 点，同时防止只拿 early 四点就误冻为“加密后”
正式图。

## 代码与输出

- plotter：`scripts/plot_bb_cost_to_precision_joint_ef_dense.py`
- pack spec freezer：`scripts/freeze_bb_joint_ef_cost_pack_spec.py`
- figure contract freezer：
  `scripts/freeze_bb_cost_to_precision_joint_ef_dense_figure_contract.py`
- tests：`tests/test_plot_bb_cost_to_precision_joint_ef_dense.py`
- 合成样式预览（绝非科学结果）：
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_ivm_vs_ode_neb_cost_to_precision_joint_ef_synthetic_preview_v2.pdf`
- 预期正式候选：
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/barnase_barstar/active_candidates/barnase_barstar_ivm_vs_ode_neb_cost_to_precision_joint_ef_dense.pdf`

样式复用 alanine/Muller 1x2 的字体、色板、空心 marker、panel 标号、图例和
箭头布局。最终图只显示 `IVM` 与 `ODE-NEB`。

## 完成数据后的精确顺序

以下命令均为后处理/校验，不会构造 OpenMM context。每个 physical evaluator
自己的 PASS、WAL 和 atomic shard 门禁必须先完成。

1. 对每个完成 bundle 生成一个包含全部点的 plotting pack spec：

```bash
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/scripts/freeze_bb_joint_ef_cost_pack_spec.py \
  --output /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/registrations/diagnostics/<PACK_SPEC>.json \
  --pack-id <PACK_ID> --method <IVM_OR_ODE-NEB> \
  --result <RESULT.npz> --audit <AUDIT.json> \
  --contract <CONTRACT.json> --manifest <MANIFEST.json> \
  --evaluator <EVALUATOR.py> --topology inputs/complex_fixed.pdb \
  --bb-core src/bb_core.py --wal <WAL1> --wal <WAL2> \
  <IVM_ONLY_REPLAY_AND_VERIFICATION_ARGUMENTS>
```

IVM-only 参数为 `--ivm-replay-result`、`--ivm-verification-report`、
`--ivm-verification-schema` 和 `--ivm-verification-status`；ODE-NEB 直接从
shard 的原始内部力重算离散 \(I_\perp\)。

2. 所有 pack 完整后冻结唯一 final figure contract：

```bash
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/scripts/freeze_bb_cost_to_precision_joint_ef_dense_figure_contract.py \
  --output /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/registrations/diagnostics/BB_JOINT_EF_COST_TO_PRECISION_FIGURE_CONTRACT_R1.json \
  --figure-output /Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/barnase_barstar/active_candidates/barnase_barstar_ivm_vs_ode_neb_cost_to_precision_joint_ef_dense.pdf \
  --pack-spec <IVM_EARLY.json> --pack-spec <IVM_LATE.json> \
  --pack-spec /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/registrations/diagnostics/BB_COST_PRECISION_PACK_ODE_EARLY_JOINT_EF_20260902_R1.json \
  --pack-spec <ODE_LATE.json>
```

3. 先只读验证，再正式渲染：

```bash
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/scripts/plot_bb_cost_to_precision_joint_ef_dense.py \
  --contract /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/registrations/diagnostics/BB_JOINT_EF_COST_TO_PRECISION_FIGURE_CONTRACT_R1.json \
  --output /Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/barnase_barstar/active_candidates/barnase_barstar_ivm_vs_ode_neb_cost_to_precision_joint_ef_dense.pdf \
  --validate-only

MPLCONFIGDIR=/private/tmp/bb_cost_joint_mpl \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/scripts/plot_bb_cost_to_precision_joint_ef_dense.py \
  --contract /Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/registrations/diagnostics/BB_JOINT_EF_COST_TO_PRECISION_FIGURE_CONTRACT_R1.json \
  --output /Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/barnase_barstar/active_candidates/barnase_barstar_ivm_vs_ode_neb_cost_to_precision_joint_ef_dense.pdf
```

4. 用 `pdfinfo`、`pdftoppm` 和逐像素目视检查做 PDF QA；最终 PDF 和同 stem
   provenance JSON 均采用 no-overwrite 原子发布。

## 当前测试覆盖

零 OpenMM tests 覆盖：可变点数 freezer-to-plotter 全链、合成 PDF/no-overwrite、
真实 pack spec freezer 的 all-points 行为、full-force shard 篡改、hash-consistent
非 arc500 坐标、错误离散 \(I_\perp\)、IVM online history 不一致、以及
diagnostic 调用误计入 optimization NoF。最终正式图仍必须在所有真实 pack
PASS 后重新运行全套测试和 PDF visual QA。
