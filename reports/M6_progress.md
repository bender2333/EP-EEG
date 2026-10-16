# M6 进度：拉长 EP 沙盘训练（epochs）

**假设**：在 M1 最佳 opts 上只增加 `epochs`（训练遍数），验证 κ 能否追上同输入 LDA（0.3889）。  
**旋钮选择**：缩放 **epochs**（默认 10 → 20 / 50 / 100），不缩放 `free_steps`/`nudged_steps`（二者是每批求解预算，不是数据遍数）。M1 搜索曾单种子试过 epochs=20（κ≈0.28）与 30（过拟合），本阶段做系统确认。

**固定设定**
- arm: `ep_track2`；opts 基线：`leak=1.0, rowmass_project=0.9, eta=0.5, temperature=0.01` + 变 epochs
- 数据：BCI Competition IV-2a / MOABB `BNCI2014_001`，9 被试，T 会话 run0–3 训 / run4–5 验；四类运动想象
- 输入：Track-2（与 M0–M5 管线一致）：每导 μ/β 功率 + 包络等 → 66 维 × 11 子窗，tanh(x/3) 压缩后入沙盘
- 硬件：8 核 CPU、无 GPU；`NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=2`，jobs=4

## 分档结果

| tier | epochs | 倍率 | κ_val | settle | wall_s | 硬件 | 判定 | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| (M1 full ref) | 10 | 1x | 0.2870 | 1.0 | 170 | 8c CPU | keep | 低于 LDA 0.3889 |
| fast | 20 | 2x | TBD | | | | | |
| fast | 50 | 5x | TBD | | | | | |
| fast | 100 | 10x | TBD | | | | | |

