---
status: proposed
---

# 组 3 五脑区导联映射与互惠边掩码（M2 前冻结）

组 3 需在 M2 前有被接受的 ADR。本条给出映射与允许边，并明确：组 1–3 共用同一静态求解、学习规则和出水口；Trace 记忆不进入主对照（移出为单独时序消融）。

## 导联归区

| 脑区 | 导联 |
| --- | --- |
| frontal | Fz, FC3, FC1, FCz, FC2, FC4 |
| left | C5, C3, C1, CP3, CP1 |
| mid | Cz, CPz |
| right | C2, C4, C6, CP2, CP4 |
| parietal | P1, Pz, P2, POz |

## 允许互惠边

- 区内全连通（微柱之间）。
- 区间：frontal↔{left,mid,right}；parietal↔{left,mid,right}；left↔mid；right↔mid。禁止 left↔right 直连与 frontal↔parietal 直连。

## 初始化与容量

- 初始化幅值与组 1 相同（同分布）；边数少于组 1，因此另设**等度随机稀疏图**对照（见 `ep_eeg.sandbox` 的 `group=sparse`）。
- 行质量与证书规则与组 1 相同。

## Consequences

- Trace / Afterglow 不进入组 3。
- 组 1 胜过组 3 时，不能单凭此判断人工脑区先验有害（边数/度分布不同）。
