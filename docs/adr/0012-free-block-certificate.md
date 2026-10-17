---
status: proposed
---

# 精化 ADR-0001 的证书：行和只计 free→free 边

ADR-0001 用 cadence 的 `certificate(graph)` 做门禁，它的 `row_mass` 取所有神经元入射有效权重绝对值之和的最大值，**单向的输入突触也计入**。M9c 的线性探针显示，这一保守计法把柱神经元的 v 窗口压在约 ±0.25（输入突触质量最大约 0.56，占去 1.8 预算的近三分之一），加上 M1 选的 `leak=1.0`（对称 tanh(v/2)），湖在全部实验中都工作在线性区（线性拟合 R² = 0.9998）。

## 决定

当输入神经元**没有任何入边**（`ep_structure(fixed_inputs=输入索引).fixed_incoming_mass == 0`）时，证书和 `project_rowmass` 的行和只计 free→free 边（col↔col、col↔motor）；输入→柱的突触既不计入，也不被投影缩放。实现为 `SandboxConfig.cert_free_only`，默认关闭，已有的臂不受影响。

## 理由（证明要点）

cadence 的单步更新为 v ← (1−dt)·v + dt·(W·act(v) + c)。输入神经元没有入边，所以状态向量按（输入，free）排列后，雅可比是**块下三角**的：

1. 输入块是 v_in ← (1−dt)·v_in + dt·c_in，以 1−dt 的速率独立收敛到由驱动决定的唯一平衡点。
2. 给定输入，free 块的映射是 v_f ← (1−dt)·v_f + dt·(W_ff·act(v_f) + W_if·act(v_in) + c_f)。其中 W_if·act(v_in) 只是外场项，对关于 v_f 的 Lipschitz 常数没有贡献；当 L·ρ_free < 1 时，它以 q_f = 1 − dt·(1 − L·ρ_free) 压缩。
3. 级联收敛：输入误差以几何速度消失，经有界增益传给 free 块，free 块自身压缩。所以全系统的平衡点唯一，收敛速率为 max(q_f, 1−dt)。

这是**更紧但仍然成立**的证书，不是放宽：它不放弃任何保证，只是不再把常数外场计进 Lipschitz 项。cadence 的 `ep_structure` 文档也写明，无入边的源神经元收敛后，它们的出边就是外场。

**注意**：cadence 用 Lean 证明的是全系统 ∞ 范数版本；这里的级联论证是标准推广，但不在那份 Lean 证明之内。`qualified=True` 的事后残差检查保留，作为第二道保险。微扰相仍不在证书覆盖之内（与 ADR-0001 相同）。

## 已核实

- 当前沙盘：`fixed_incoming_mass = 0.0`，`free_asymmetry = 0.0`，没有 adaptation。
- 初始化时：全图 row_mass 为 1.272，其中输入贡献最多 0.563，free→free 最多 0.889。
- σ_in × 4 / 8 / 16 时，全图证书不通过（ρ_full = 2.96 / 5.22 / 9.72），精化证书通过（ρ_free = 0.889）。

## 适用范围

只用于 M9c 的最终实验（非线性湖上的冲刷检验）。在 M0–M9c 的已有结果中，输入突触一直计入行和，这些结果不受影响。
