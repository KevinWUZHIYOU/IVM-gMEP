# Chignolin 等距规范 IVM--NEB 正式报告（v52）

## 结论

正式 IVM 运行是
`ivm_equal_idpp_v52_direct_then_certified_fallback_production`。它与固定
`k=50`、仅 ODE 的 NEB 使用逐元素完全相同的 22-image IDPP 初始路径，
共同 SHA-256 为
`57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996`。

IVM 在 500 max joint evaluations / internal image 时得到：

- 未归一化 \(I_\perp=0.0381150078\)；
- 稀疏 image 能垒 \(=19.6904470\) kcal/mol；
- 独立加密 0 K 能垒 \(=20.3754058\) kcal/mol；
- physical normal \(f_{\max}=0.00789048\) eV/Å；
- 等距积分
  \(\int_0^1(\phi'\!\cdot\!\phi'')^2\,ds
  =5.91484\times10^{-20}\)；
- segment-length CV \(=8.03611\times10^{-16}\)，最大/最小段长比
  \(=1.000000000000004\)。

NEB 在 5000 calls/image 时的 \(I_\perp=0.0350337\)，但需要 4124 calls
才首次达到 IVM 的 \(I_\perp=0.0381150\) 阈值；IVM 首次达到该阈值是
344 calls，因此加速为 \(11.99\times\)。在双方都实际达到的加密能垒
阈值中，最严格的共同阈值是 20.6416317 kcal/mol：IVM 需要 275 calls，
NEB 需要 3900 calls，对应 \(14.18\times\) 加速。全部共同能垒阈值的
加速范围为 \(12.5\)--\(16.0\times\)。

IVM 在 call 300 的最低加密能垒为 20.3566704 kcal/mol；call 344
进一步降低 \(I_\perp\) 后，加密能垒轻微回升约 0.019 kcal/mol。
图和成本统计均保留这一真实波动，成本使用标准的 first-passage
running minimum。NEB 在 5000-call 轨迹中的最低加密能垒为
20.4076414 kcal/mol（call 4250），因此低于该值的 IVM 阈值标为
NEB 未达到，不外推也不进入双方法 cost-to-threshold 曲线。

## 为什么过去看起来无法同时降低 \(I_\perp\) 和弹簧残差

连续 string 理论中两者并不矛盾：法向运动改变几何路径，切向运动只
改变参数化。冲突来自 22-image 离散化和旧的接受方式。

1. 离散中心差分 tangent
   \[
   \tau_i=\frac{x_{i+1}-x_{i-1}}
   {\|x_{i+1}-x_{i-1}\|}
   \]
   会随相邻 image 的非均匀间距发生旋转。因此旧路径的一部分低
   \(I_\perp\) 实际依赖不均匀采样；事后重参数化会暴露新的离散法向力。
2. 弹簧移动是全路径耦合操作。若重参数化后逐 image 独立接受或回退，
   会制造交替的短段/长段。等距积分包含类似二阶差分的项，会强烈放大
   这种高频误差。
3. 旧算法先完成物理步、再对已经改变的 tangent 做有限幅度重参数化，
   但没有对重参数化后的整条 band 重新认证同一个 IVM 子问题。

解决方法不是在两个指标之间加经验权重，而是把等距条件当作参数化
规范，并把接受/回退提升到整条 band。

## v52 算法

每个外层 IVM 步执行以下操作。

1. 用统一的中心差分冻结整条路径的 tangent。
2. 在冻结法向空间中求解真正的 proximal IVM 子问题
   \[
   \min_{\delta_i\perp\tau_i}
   E(x_i+\delta_i)+\frac{\|\delta_i\|^2}{2\alpha}.
   \]
   搜索方向来自 projected L-BFGS；线搜索使用精确 proximal objective，
   相对 KKT forcing tolerance 固定为 0.8。
3. 在请求 OpenMM 之前，把整条候选 band 投影到 central-Hermite
   曲线上的相同弹簧长度平衡。该投影只使用坐标和中心差分几何，不调用
   能量或力。切向变化是 gauge reparameterization，不改变 IVM 的法向
   物理目标。
4. 只在最终投影坐标上联合计算能量和力。只有当所有 image 的精确
   proximal objective 均不增加、相对 KKT 合格且刷新后的全局
   \(I_\perp\) 不增加时，才同步提交整条 band；否则整条 band 回退。

这个 whole-band transactional 规则消除了“某些 image 接受、相邻 image
回退”产生的锯齿形间距误差。每个已提交的直接求解器迭代都已经在
equal-spring gauge 上，所以不需要事后再用 OpenMM 力做重参数化。

累计曲率只保留在同一个冻结 tangent 的子问题内部。跨外层迭代复用旧
L-BFGS secant pair 的候选方案没有选入正式算法，因为 tangent 旋转后
旧曲率属于不同的法向空间，会降低接受稳定性。v52 因此没有使用按调用
次数硬切换的 warm-up history。

## \(\alpha\) 策略与无硬调阶段

\(\alpha\) 的初值为 1，允许范围为 \([0.03,20]\)。控制器只依据实际
proximal objective/model quality、KKT 和接受结果作 shrink/hold/grow，
不读取调用次数，也没有人为的 early/mid/late schedule。

在直接 equal-spring 子问题阶段，所有通过认证的步都使控制器保持
\(\alpha=1\)。call 324 后，完整直接子问题没有可接受步，于是触发一个
状态条件：永久退出快速 predictor，转入同一 proximal IVM 目标的保守
认证求解器。随后 \(\alpha\) 从 1 缩至 0.5，并在 call 344 接受最后一个
改进，把 \(I_\perp\) 从 0.0388524 降至 0.0381150。后续不合格候选使
\(\alpha\) 继续收缩到下限 0.03，但没有改变已发布路径。所有这些真实
候选仍计入 500-call 横轴。

## 全阶段 NEB 包络审计

审计包含 14 条与 IVM 具有同一初始路径哈希的 NEB 轨迹。对 ODE-NEB
偶尔跳过的整数最大计数标签，只使用该预算或之前最后一个已保存状态，
从不使用未来状态回填。

在 calls 2--500：

- \(I_\perp\) 相对 NEB 逐 call 最优包络没有任何失败点；
- 稀疏能垒相对同一包络没有任何失败点；
- calls 2--50 的最小 NEB/IVM 比为 \(2.408\times\)（\(I_\perp\)）
  和 \(1.991\times\)（能垒）；
- calls 51--200 的最小比分别为 \(35.651\times\) 和
  \(1.391\times\)；
- calls 201--500 的最小比分别为 \(11.767\times\) 和
  \(1.170\times\)。

因此 v52 修复了 v46 在最早几个调用点落后 NEB 包络的问题；领先不是由
拼接调参曲线获得，而是来自单条 1--500 calls 的正式生产轨迹。

## 等距结果

IVM 的等距积分在整个正式轨迹中处于浮点数值零附近，最终为
\(5.91\times10^{-20}\)。固定 NEB 的等距积分在 call 500 为
338.223，在 call 5000 仍为 44.7224；对应 segment CV 分别约
\(5.68\times10^{-4}\) 和 \(1.60\times10^{-4}\)。因此 v52 不只是把
等距积分降到 NEB 的同一数量级，而是把弹簧残差降到了数值精度限制。

## 加密能垒

每一条被选中的路径都对 21 个相邻 image 段先做 10 和 20 等分线性
加密，并用独立 diagnostic evaluator 计算真实 OpenMM 0 K 能量。
若两级能垒差超过 0.05 kcal/mol，才继续到 40 和 80 等分。v52 的全部
20 个诊断快照在 20 等分即通过 0.05 kcal/mol 标准。

v52 dense grid 的 5868 次 diagnostic OpenMM 调用与 cache miss 严格
相等；NEB dense grid 为 38550 次。两者都与优化横轴完全分离。

## 联合能量/力调用审计

每个 image 使用独立 evaluator。实验内保存所有精确坐标及其联合
energy/force 结果；回到字节完全相同的旧坐标时命中历史缓存。缓存不在
不同实验之间共享。

IVM 的 20 个内部 image 最终计数
min/mean/max/sum 为 \(359/370.35/500/7407\)。包括两个端点后共 7409
次底层 OpenMM 联合评估，严格等于 22 个 evaluator 的 cache miss 之和。
NEB 的对应内部计数为 \(5000/5000/5000/100000\)，包括端点共 100002。

IVM 有 176 个“保存路径不变、真实调用仍增加”的区间。这直接证明被拒绝
或回退的候选没有从计数中消失。energy API requests、force API requests、
cache hits 和 joint evaluations 均保留完整的逐 image 向量。

## 验证

- 正式 validator 无失败项；
- 48 个自动测试全部通过；
- IVM 与 NEB 初始路径逐元素相同；
- 所有生产计算和诊断只声明并使用中心差分 tangent；
- 输出中不存在 `normalized_i_perp`；
- 源码扫描未发现 ASE improved tangent；
- checkpoint 能恢复实验内历史缓存和累计计数；
- 最终曲线只来自单条 v52 正式生产运行。
