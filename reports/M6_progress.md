# M6 进度：拉长 EP 沙盘训练（epochs）

**假设**：在 M1 最佳 opts 上只增加 `epochs`（训练遍数），验证 κ 能否追上同输入 LDA（0.3889）。  
**旋钮选择**：缩放 **epochs**（默认 10 → 20 / 50 / 100），不缩放 `free_steps`/`nudged_steps`（二者是每批求解预算，不是数据遍数）。M1 搜索曾单种子试过 epochs=20（κ≈0.28）与 30（过拟合），本阶段做系统确认。

**固定设定**
- arm: `ep_track2`；opts 基线：`leak=1.0, rowmass_project=0.9, eta=0.5, temperature=0.01` + 变 epochs
- 数据：BCI Competition IV-2a / MOABB `BNCI2014_001`，9 被试，T 会话 run0–3 训 / run4–5 验；四类运动想象
- 输入：Track-2（与 M0–M5 管线一致）：每导 Hjorth（activity/mobility/complexity）→ 66 维 × 11 子窗，tanh(x/3) 压缩后入沙盘
- 硬件：8 核 CPU、无 GPU；`NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=2`，jobs=4

## 分档结果

| tier | epochs | 倍率 | κ_val | settle | wall_s | 硬件 | 判定 | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| (M1 full ref) | 10 | 1x | 0.2870 | 1.0 | 170 | 8c CPU | keep | 低于 LDA 0.3889 |
| fast | 20 | 2x | 0.2824 | 1.0 | 62 | 8c CPU | keep | 与 M1 fast 相当 |
| fast | 50 | 5x | 0.3272 | 1.0 | 149 | 8c CPU | keep | 高于 1x；仍低于 LDA |
| fast | 100 | 10x | 0.3349 | 1.0 | 297 | 8c CPU | keep | 略高于 5x；收益递减 |

### 阶段性通报（fast 2x / 5x 完成）

1. **数据**：BCI IV-2a（BNCI2014_001），9 被试，T 会话 train run0–3 / val run4–5，四类 MI。
2. **输入**：Track-2 Hjorth 66 维 × 11 子窗；EOG 回归 + 4–35 Hz + Z-score + FeatureScaler + tanh(x/3)；与 M0–M5 一致。
3. **墙钟**：2x=62s；5x=149s（8 核 CPU、无 GPU）。
4. **κ**：2x=0.2824；5x=0.3272（目标 LDA=0.3889；1x full=0.2870）。5x 有提升，未追上 LDA。

### 阶段性通报（fast 10x 完成）

1. **数据**：同上（IV-2a / BNCI，9 被试，T 训验切分）。
2. **输入**：同上（Track-2 Hjorth 66×11）。
3. **墙钟**：10x=297s（8 核 CPU、无 GPU）。
4. **κ**：10x=0.3349。相对 5x 仅微升；均未追上 LDA 0.3889。下一步：完整档 9×3 确认 2x/5x/10x。

### 阶段性通报（full 2x / 5x 完成）

1. **数据**：IV-2a / BNCI2014_001，9 被试 × 3 种子，T 训验切分。
2. **输入**：Track-2 Hjorth 66×11，与既有管线一致。
3. **墙钟**：2x full=145s；5x full=358s（8 核 CPU、无 GPU）。
4. **κ**：2x=0.2994 (sd=0.0187)；5x=0.3292 (sd=0.0018)。均未追上 LDA 0.3889。10x full 进行中。

### 阶段性通报（full 10x 完成 → 结项）

1. **数据**：IV-2a / BNCI2014_001，9×3，T 训验。
2. **输入**：Track-2 Hjorth 66×11（冻结管线）。
3. **墙钟**：10x full=697s（8 核 CPU、无 GPU）。
4. **κ**：10x full=0.3354 (sd=0.0009)。**未追上 LDA 0.3889**。见 `reports/M6.md`。
