# Chignolin：Static-NEB、Original String、OBSM 与 IVM/ODE-NEB 对比

本报告冻结 2026-07-27 完成的 Chignolin 方法对比。所有最终曲线均来自独立生产运行；没有将调参轨迹拼接成正式曲线。

## 共同协议

- 20 个内部 image，加两个固定端点，共 22 个 image。
- 五种方法逐元素复用同一条 IDPP 初始路径：
  `57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996`。
- 所有优化力、约束子问题、重参数化诊断和绘图诊断统一采用
  \[
  \tau_i=\frac{x_{i+1}-x_{i-1}}{\|x_{i+1}-x_{i-1}\|}.
  \]
- OpenMM 使用 CPU double precision、`amber14-all.xml + implicit/gbn2.xml + NoCutoff`。
- 不使用 climbing image。
- \(I_\perp\) 为未归一化量。
- 0 K forward barrier 以 folded 端点能量为零点。每段先做 10/20 等分；差异超过
  0.05 kcal mol\(^{-1}\) 时继续至 40/80 等分。

## 横坐标：整条路径 Number of Force Evaluations

所有含计算代价的图使用

```python
number_of_force_evaluations = counter_vectors.sum(axis=1)
```

这是真实联合 energy-force OpenMM cache miss 在全部 22 个 image 上的总和。一次后端评估同时产生 energy 和 force，只计一次；同一坐标的重复 energy/force 请求命中缓存，不增加计数。固定端点的首次评估也计入，因此均匀完成 500 次内部 image 评估时通常为
\(20\times500+2=10002\) 次。

稠密 barrier 使用独立 diagnostic evaluator；其调用量另行审计，不进入优化横轴。

## 最终方法

| 方法 | 独立生产设置 | 最终 \(N_F\) |
|---|---|---:|
| IVM | `state_ivm`，固定 proximal \(\alpha=1\)，每个冻结子问题最多 15 次内部联合评估，state-curve projected reparameterization | 6823 |
| ODE-NEB | ASE ODE optimizer，中心差分 NEB，\(k=50\)；正式比较只取前 500 calls/image | 10002 |
| Static-NEB | 固定步长显式 Euler 的中心差分 NEB，\(k=35,\alpha=0.0125\) | 10002 |
| Original String | 每个内部 image 用完整物理力做原生 steepest-descent，\(\Delta t=0.0125\)，随后每步做 Appendix-A cubic-spline 重参数化 | 10002 |
| OBSM-FIRE | 冻结中心差分 normal hyperplane；bare-\(U\) FIRE 子优化；\(\lambda=0.9\)，10 个 inner steps，\(\Delta t_{\rm FIRE}=0.12,\beta=1\)，每个 outer step 做 cubic-spline 重参数化 | 9982 |

Original String 的更新是
\[
x_i^{n+1/2}=x_i^n+\Delta t\,F_i(x_i^n),
\]
没有 tangent projection；只有随后重参数化负责重新分布 image。

Original String 与 OBSM 使用的 cubic spline 和 alanine 正式实现逐函数完全相同。它是 Simplified String Method (2007) Appendix A 的手写实现：解 \(\sigma\) 的对称三对角系统，两个端点的三阶导数由首尾各四点的插值三次多项式给定，然后在均匀弦长标签上求值。重参数化本身不请求 OpenMM energy/force。

## 调参规模与选择

### Static-NEB

- 宽扫描：156 个 \((k,\alpha)\)，100 calls/image；125 个稳定。
- 稳定边界加密：73 个配置，300 calls/image；69 个稳定。
- 总计 229 个配置。
- 选择 \(k=35,\alpha=0.0125\)：在终点 \(I_\perp\)、barrier 和等距积分之间给出最好的稳定 whole-curve 折中。

### Original String

- 宽扫描：33 个 \(\Delta t\)，150 calls/image；29 个稳定。
- 稳定边界加密：39 个 \(\Delta t\)，300 calls/image；35 个稳定。
- 总计 72 个配置。
- \(\Delta t=0.0125\) 是最优稳定点；\(\Delta t=0.0128127\) 已进入不稳定区。

### OBSM-FIRE

- 宽扫描：180 个配置，覆盖 \(\lambda\)、inner steps 与 FIRE \(\Delta t\)。
- 中等步长扩展：147 个配置。
- 高步长稳定边界：70 个配置。
- \(\beta\) 与 reparameterization interval 加密：180 个配置。
- 总计 577 个调参配置，另有 6 个独立 500-call finalist 生产运行。
- \(\Delta t=0.12\) 全部稳定；0.13 全部稳定；0.14 仅部分稳定；\(\Delta t\ge0.15\) 全部不稳定。
- 最终不是机械采用单一加权排名，而是在 6 条独立 finalist 中选择
  \(\lambda=0.9\)、10 inner steps、\(\Delta t=0.12\)、\(\beta=1\)、每步重参数化的 Pareto 候选。它的终点 \(I_\perp\) 和稠密 barrier 均低于 ODE-NEB，并在中后期 first-passage 代价上占优。

所有调参只用于配置选择；正式曲线来自选择完成后的单条独立生产运行。

## 最终结果

| 方法 | 最终 \(N_F\) | \(I_\perp\) | 稠密 \(\Delta E\) (kcal mol\(^{-1}\)) | physical \(f_{\max}\) (eV Å\(^{-1}\)) | \(\mathcal I_{\rm eq}\) |
|---|---:|---:|---:|---:|---:|
| IVM | 6823 | 0.038852 | 20.373740 | 0.008063 | \(5.89\times10^{-20}\) |
| ODE-NEB | 10002 | 0.464218 | 25.490265 | 0.022806 | 338.223 |
| Static-NEB | 10002 | 1.867610 | 29.841996 | 0.050892 | 408.097 |
| Original String | 10002 | 1.869955 | 29.789349 | 0.050950 | 0.525085 |
| OBSM-FIRE | 9982 | 0.424642 | 24.430799 | 0.072057 | 1858.578 |

离散 checkpoint 的 first-passage 结果如下，不做曲线插值：

| 目标 | IVM \(N_F\) | 最快非 IVM 方法及 \(N_F\) | 加速 |
|---|---:|---:|---:|
| \(I_\perp\le100\) | 142 | ODE-NEB, 882 | 6.21× |
| \(I_\perp\le10\) | 202 | ODE-NEB, 2222 | 11.0× |
| \(I_\perp\le1\) | 502 | OBSM-FIRE, 5302 | 10.56× |
| \(I_\perp\le0.5\) | 762 | OBSM-FIRE, 9482 | 12.44× |
| \(\Delta E\le100\) kcal mol\(^{-1}\) | 82 | OBSM-FIRE, 462 | 5.63× |
| \(\Delta E\le50\) kcal mol\(^{-1}\) | 162 | OBSM-FIRE, 682 | 4.21× |
| \(\Delta E\le30\) kcal mol\(^{-1}\) | 362 | OBSM-FIRE, 2662 | 7.35× |
| \(\Delta E\le27\) kcal mol\(^{-1}\) | 602 | OBSM-FIRE, 4862 | 8.08× |
| \(\Delta E\le26\) kcal mol\(^{-1}\) | 902 | OBSM-FIRE, 6182 | 6.85× |

能垒 cost-to-precision 图使用“达到给定稠密 0 K barrier 阈值所需的最少 \(N_F\)”；不再使用依赖任意参考值的
\(|\Delta E-\Delta E^*|\)。IVM 曲线只画到真实停止点，不外推。

## 固定 \(\alpha\) 灵敏度

固定其余 IVM 算法与配置，独立比较
\(\alpha=0.03,0.1,0.3,1,3,10,20\)。在 \(N_F\le1002,3002,6002\)
三个预算上，\(\alpha=1\) 均同时给出最低 \(I_\perp\) 和最低稠密
barrier：

| \(N_F\) | \(I_\perp\), \(\alpha=1\) | 稠密 \(\Delta E\), \(\alpha=1\) (kcal mol\(^{-1}\)) |
|---:|---:|---:|
| 1002 | 0.323773 | 24.770558 |
| 3002 | 0.096807 | 22.727157 |
| 6002 | 0.046017 | 20.356670 |

因此采用固定 \(\alpha=1\) 不是从时间表中逐段硬调出来的选择，而是同一算法的独立固定参数敏感性结果。

## 图片

所有图均提供 PDF、SVG 和 600 dpi PNG，主目录为
`figures/paper_methods_total_nf_20260727a/`：

- `chignolin_main_metrics_total_nf`
- `chignolin_cost_to_precision_total_nf`
- `chignolin_speedup_vs_ode_total_nf`
- `chignolin_method_diagnostics_total_nf`
- `chignolin_force_call_audit_total_nf`
- `chignolin_static_neb_k_alpha_sensitivity`
- `chignolin_original_string_dt_sensitivity`
- `chignolin_obsm_fire_dt_sensitivity`
- `chignolin_obsm_beta_reparam_sensitivity`
- `chignolin_ivm_fixed_alpha_sensitivity_total_nf`

每张图的数值表位于同目录的 `data/`。

## 审计与测试

- 五条正式曲线的初始路径 hash 完全一致。
- 所有正式运行声明并实际使用中心差分 tangent。
- 每条生产运行均满足底层 OpenMM 联合调用数等于 22 个 image evaluator 的 cache-miss 总和。
- force→energy、energy→force、重复请求、坐标回退、checkpoint 恢复和 run-local 历史缓存均有自动测试。
- Original String 的纯切向力分量保留测试通过。
- Appendix-A spline 的节点插值、三次多项式精确复现、端点边界和端点保持测试通过。
- 拒绝/回退调用不会从计数中消失。
- diagnostic barrier 调用与优化横轴完全分离。
- 最终完整测试为 53/53 通过。
- 10 个 PDF 均为单页且字体嵌入；10 个 SVG 均可被 XML 解析；PNG
  均为约 4200--4400 像素宽的 600 dpi 输出。PDF 经 Poppler 渲染后逐张检查，无裁切、重叠或缺字。

## 重建

在 Chignolin 根目录：

```bash
OPENMM_CPU_THREADS=1 PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/run_selected_method_productions.py --root .
```

```bash
OPENMM_CPU_THREADS=1 PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/run_selected_dense_diagnostics.py --root .
```

```bash
PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/plot_paper_method_comparison.py --root .
```

固定 \(\alpha\) IVM 灵敏度：

```bash
OPENMM_CPU_THREADS=1 PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/run_ivm_fixed_alpha_sensitivity.py --root . --budget 500 --workers 6
```

```bash
OPENMM_CPU_THREADS=1 PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/run_ivm_alpha_dense_diagnostics.py --root .
```

```bash
PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/plot_ivm_fixed_alpha_sensitivity_total_nf.py --root .
```

完整测试：

```bash
OPENMM_CPU_THREADS=1 PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python tests/run_tests.py
```
