---
status: proposed
---

# M5 出水口扩展：先做多槽，连续目标等人确认来源

AGENTS.md 与 plan 将「连续目标来源」保留给人决定：IV-2a 没有连续轨迹标签，不得伪造。

## 本阶段执行范围（保守）

1. **冻结** M2 组 1 沙盘结构与超参（leak/rowmass/eta/temperature/Track-2）。
2. **只更换出水口配置**：`slots=1`（基线，与 M2 ep_g1 相同）对照 `slots=4`（`Learner(slots=4)`，四类各一槽）。
3. **不做** `nudge="quadratic"` 连续向量目标，也不断造伪连续标签。

## 若人确认连续目标来源后

另开阶段或接受本 ADR 的修订，再引入真实连续标签协议与对应基线；彼时才允许 `nudge="quadratic"`。

## Consequences

- M5 完成标准中「连续目标来源已由人确认」在本自动推进中记为**未满足、待决**；多槽对比仍可完成并报告。
- 不把 one-hot 二次型或任意平滑插值当作「连续目标」。
