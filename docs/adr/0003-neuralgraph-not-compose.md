---
status: accepted
---

# 主线直接用 Genome / NeuralGraph / Learner 搭建，不用 Brain.compose

cadence 主推的接口是 `Brain.compose` + `live`，本项目不采用，原因有三：

- `compose` 的 `modules` 是一条串联的稠密区域链，表达不了 [ADR-0002](0002-vertically-anchored-mesh.md) 的锚定拓扑。
- 它自带 prefrontal 工作迹、联想记忆和 arousal，会给组间对照引入额外的变量。
- 它的默认组合图不满足收敛证书：`Brain.compose(8, 2)` 的 row_mass 约为 11.5，而限值是 2。这与 [ADR-0001](0001-static-subwindow-equilibrium.md) 冲突。

因此主线用 `Genome` 或 `Connectome.from_synapses` 建图，用 `NeuralGraph` 求解，用 `Learner` 学习，由本仓库的适配层包装。

## Consequences

- **compose 原本代办的事要自己处理**：出水口的侧抑制（低层接口默认没有）、全图宽度的 drive 组装（`Learner.step` 要求形状为 (batch, conn.n)）、保存与加载（`Learner.save` 只保存 learner）。
- **步长不同**：`learning_neuron_model()` 的默认 dt=0.5，L=0.5，证书要求 row_mass < 2。compose 用的是 dt=1.0。
- **声明不外推**：按 cadence 自己的定位，这条路径属于"隔离的分类/机制对照"，不是它主打的连续大脑生命周期。本项目的结论不外推为 cadence 连续大脑的能力。

来源：讨论 02、03 的示例代码都走这条路径，但没有说明理由；理由与数值已对照 cadence `6499e56` 核实。
