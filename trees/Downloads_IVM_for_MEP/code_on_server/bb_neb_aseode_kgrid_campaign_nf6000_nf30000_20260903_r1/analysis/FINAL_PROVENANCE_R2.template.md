# Barnase--Barstar 离散 32 图像、ODE-NF30000 比较图溯源

创建：@CREATED_HKT@

状态：**FINAL；全部输入、离散指标、输出和 PDF 验收均由 completion audit 闭包。**

## 登记对象

@OUTPUT_TABLE@

最终输入清单：@MANIFEST_BINDING@

完成审计：`@COMPLETION_AUDIT_PATH@`。

## 科学口径

两项主要指标来自同一条已保存的 32 图像路径、同一组 32 点能量和完整 Cartesian 力：

\[
t_i=\frac{x_{i+1}-x_{i-1}}{\lVert x_{i+1}-x_{i-1}\rVert},\qquad
I_\perp=\sum_{i=1}^{30}\frac{\ell_{i-1}+\ell_i}{2}
\left\lVert F_i-(F_i\cdot t_i)t_i\right\rVert^2,
\]

\[
\Delta E_{32}=\max_{0\le i\le31}E_i-E_0.
\]

这里不做路径加密；tangent 与 IR-IVM 一样采用中心差分。固定端点已由登记缓存继承，优化
Number of Forces 贡献为 0；每个 ODE paid batch 计 30 个内部图像的联合能量/完整力调用，
包括被拒绝的 trial。ODE-NEB 使用筛选选出的 `k=@SELECTED_K@`，正式历史到 NF30000。

正式 `IVM` 曲线仅使用最后正式提交的 call 318 / NF8982。call 515 / NF14983 是算法返回的
terminal uncommitted charged tail，只能在能量--弧长和 evolution 图中以 diagnostic 明确标注，
不进入 cost-to-precision、同 NoF 表或正式 IVM 性能结论。

因此，cost-to-precision 图严格只含正式 IVM（终点 NF8982）与 ODE-NEB（终点 NF30000）
两种方法；energy-vs-arc-length 图严格包含四条由已保存数组直接得到的 32 点折线：正式
IVM / NF8982、ODE-NEB / NF30000、Previous IVM / NF9990，以及明确标注为
uncommitted diagnostic 的 IVM trial / NF14983。

@METRIC_SUMMARY@

## ODE-NEB 与本机零物理缓存

ODE 终态审计：@ODE_TERMINAL_BINDING@

离散指标缓存：@CACHE_BINDING@

本机 Mutagen 副本只读适配器与独立收据：@CACHE_TRANSPORT_BINDINGS@

缓存从第 1 批重建；两份门禁冻结前生成的缓存明确标记为
`PREAUDIT_DIAGNOSTIC_NOT_FINAL`，未被继承。缓存逐批要求完整换行 WAL 发布记录、匹配的
optimizer commit、文件 SHA-256 以及 full/XOR 重建哈希闭合。缓存和本登记过程均为 0 次物理调用。

## IR-IVM 与能量面

IR 状态绑定：@IR_BINDINGS@

evolution 使用围绕 call 515 / NF14983 返回终态构造的真实 `59x28` 未松弛 Cartesian 0 K
截面；它是诊断背景，不把 call 515 改写为正式 IVM 结果。能量面及中心语义绑定：
@SURFACE_BINDINGS@

原始能量面数据完全保留。极高 steric-clash 区只在绘图副本中按登记上限裁切颜色显示，
不得改变任何存储能量、路径或指标。

evolution 的共同 NoF 时刻通过终态前 protocol amendment 登记为
`2040, 15000, 30000`。原拟用的 NF240 中，IVM 完整路径仍在新截面内
（法向范围 `0--0.6275520851946432 Å`），但 ODE-NEB 完整路径达到
`0.7038122887583026 Å`，超过截面上界 `0.69 Å` 共 `0.0138122887583026 Å`；
因此不裁路径、不扩造能量面。NF2040 是已保留且几何有效的里程碑，选择发生在
ODE-NF30000 结果产生前，并且它更接近旧图首个松弛后比较 panel 的实际
IVM/ODE-NEB NoF `2792/2852`；它**不是按性能挑选**，也不声称属于原始实验预注册。
另一个更早的几何可用共同预算 NF990 也已披露：IVM 实际 NF969 的法向范围为
`0--0.4513690503490503 Å`，ODE-NEB NF990 为 `0--0.640372983587181 Å`，两者均在截面内。
未采用 NF990 的唯一理由是 NF2040 更接近旧图首个松弛后 panel 的 NoF；两者的取舍都不使用
性能指标。
NF30000 终态仍须通过同一严格包络断言，否则 evolution 正式图拒绝生成。

旧图 `Figures/barnase_barstar/barnase_branch_local_true0k_evolution.pdf` 只提供
`1x4`、共享坐标轴和右侧 colorbar 的布局语言。旧图的 `0,50,200,500` 是 checkpoint
标签而非 NoF；对应三个比较 panel 的实际 IVM/ODE-NEB NoF 分别为
`2792/2852, 11309/11402, 28073/28502`，不得移植为本次成本轴。

## 代码、机器附件与 PDF 验收

@CODE_TABLE@

机器可读比较 JSON、同 NoF CSV、三张 PDF、最终输入清单、PDF-QA、surface、ODE terminal/cache、
适配器/收据及上述代码均由 completion audit 记录绝对路径、SHA-256 和字节数。

PDF 验收：@PDF_QA_SUMMARY@

所有生成、验证与登记步骤均声明 `physical_calls=0`，且未加载 OpenMM、未获取 Barnase--Barstar
计算锁、未修改论文根 `main.pdf`。
