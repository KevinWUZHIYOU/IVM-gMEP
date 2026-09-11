# 启用代码快照

本目录按科学职责分为四类：

- `ivm/`：最终 fixed `alpha=20` Stage-IV IVM 核心、two-phase wrapper、
  Pareto archive 构建与 fail-closed 审计；
- `baselines/`：Original String、ASE Static/ODE NEB 入口、OBSM--BFGS/FIRE、
  调参调度和 bonded-H0/力场预处理；
- `plotting/`：`main.tex` 实际采用的三张 Ala 图的最终 plotter
  及其必需的读取/绘图依赖；另含 2026-09 新增的
  `plot_ala_delta_aware_geometric_action_1x3.py`，用于已登记到独立
  `paper0819/main.tex` 的 geometric-action 三联图，以及
  `plot_ala_grid_ivm_path_evolution_1x4.py`，用于同一被选 IVM/ODE--NEB
  实验的 Ramachandran 路径演化四联图；
- `metrics/`：2026-09 新增的离散 geometric-action evaluator
  `postprocess_ala_geometric_action.py` 及其单元测试。它们对冻结优化路径做
  离线诊断，不把额外 OpenMM 调用加入优化 NoF。

除上述明确标注的 2026-09 诊断资产外，这些文件是 2026-08-04 从两个可执行实验
谱系复制的冻结审阅快照。
为了避免最终 plotter 中的仓库根路径定位失效，实际重跑时使用：

- IVM 与三个最终 plotter：
  `/Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801/src/`；
- baseline dynamics 与 legacy plotting dependency：
  `/Users/wuzhiyou/IVM_for_MEP/python/alanine_v10_chignolin_v9_fixed_alpha_20260730/src/`。

快照不是另一份可独立修改的源码。若对核心算法、acceptance、计费或 plotter
做任何变更，应先修改可执行源谱系、重跑审计，再整体刷新本快照和 `SHA256SUMS`。
