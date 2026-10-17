# reports 索引与阶段归类

- **M0–M5**：原始协议阶段（合格交付物）。E 会话只在阶段门锁定后评估一次，记录在 holdout.tsv。
- **M6–M8**：「机制探索与失效归因阶段」，不作为 M0–M5 协议的合格交付物。目标是机制定性与解除瓶颈。
- M8：协议切分 t04_val45，不读 E；跨被试共享臂（triple）已停用。

## E 评估去重（holdout.tsv: e_eval_validity）
| 条目 | 状态 | 对应有效评估 |
|---|---|---|
| M1__ep_track2 | 有效 | — |
| M2__ep_g1 | 配置重跑（重复） | M1__ep_track2 |
| M5__ep_g1 | 配置重跑（重复） | M1__ep_track2 |
| M1__ep_fixed | 有效 | — |
| M2__ep_fixed | 配置重跑（重复） | M1__ep_fixed |
