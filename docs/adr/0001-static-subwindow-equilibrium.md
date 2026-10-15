---
status: accepted
---

# 阶段一不跨子窗延续状态：每个子窗独立求合格的静态平衡

方案原设想在一个试次的滑窗之间保留神经状态（Carried BrainState），让上一时刻的"水波余韵"延续到下一时刻。可是在满足收敛证书（`certificate(graph).certified`，即压缩映射 rate < 1）的图上，给定驱动下的不动点是唯一的，热启动和冷启动会收敛到同一状态（差别不超过停止容限），延续下来的状态会被松弛过程抹掉。要获得真正的状态记忆，就得放弃证书、接受多个能量盆地，而那样 EP 的正负微扰可能落进不同的盆地，对比梯度会失真。

因此阶段一的做法是：每个子窗独立求自由相和微扰相的合格平衡（`qualified=True`），时间信息只通过输入边界上的子窗特征进入（见 [ADR-0004](0004-subwindow-dual-track-encoding.md)）。需要时序的版本留到以后的阶段，用 `TemporalPatchNet` 对整段路径联合求解。

## Considered Options

- **有限步松弛追踪**（`qualified=False`）：训练出来的不再是 EP，而是一个截断松弛的 RNN，`contrast_asymmetry` 一类诊断也随之失去意义。这条路只作为消融。
- **Afterglow / Trace 残影注入输入**：它们都带 decay 常数，没有去掉方案本想去掉的人工衰减率。
- **放弃证书，依赖多稳态磁滞**：理由见上。

## Consequences

- 证书只覆盖自由相。微扰相另需满足 L·(row_mass + β·K_C) < 1，其中 K_C = 1/(2T)（见 cadence `docs/learning.md`）。
- 学习会抬高 row_mass，而 `scale_cap` 只裁剪单个突触的效能，不约束行和。所以初始化时和每个 epoch 结束时都要重查证书。
- `qualified=True` 时，微扰相只有 `nudged_steps` 的预算，没有回退。这个预算要与 `free_steps` 相当（cadence 文档的示例是 512/512），否则每次学习都会抛出 `LearningPhaseError`。
- `TemporalPatchNet` 是独立的 A/B/C 模型，不是 NeuralGraph。要在它上面保留 [ADR-0002](0002-vertically-anchored-mesh.md) 的拓扑，需要用 `cadence.experimental.PartitionedTemporalPatchNet` 的掩码来表达。

来源：`docs/discussions/` 01 雷区 1、02 雷区 1、03 第一节；已对照 cadence `6499e56` 核实。
