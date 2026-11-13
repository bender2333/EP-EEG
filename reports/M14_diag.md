# M14 诊断：池到出水口缺一条跳连，还是沙盘里丢掉的

探索性。不改 `reports/M14.md` 的预注册判据，不改 `receipts/M14/selected.json` 里的选定 (η, U)，不改冻结评估层，不读留出集。下面的 0.05 只用来给这一次诊断写读法，不能回头改 M14 的结论。

## 1. 假设

M14 里臂 C（平均参考前端 + 近纯平方 + 时间平均池 → 22×4 微柱的沙盘 → 4 个奇出水口）比 C_nolake（同一套池，出水口直接读池）低大约 0.10 κ。代码上的差别是：`lake=True` 时出水口只和微柱互惠，不和 22 个池神经元相连（`ep_eeg/m14.py` 建图处）。C 的训练准确率也低于 C_nolake（报告 M14 §3.4：C 0.407，C_nolake 0.574），所以不像是只在验证集上过拟合。

本诊断只动一件事，并另做一次前向探针：

1. **探针**：在自由相平衡点上，用收缩 LDA 读池、微柱、出水口各自的试次状态。只回答信息在哪一层还在。
2. **C_skip**：与 C 相同，另加池 ↔ 出水口的互惠投射，单独一条初始化预算 `skip`。只回答「缺的这条跳连是不是充分原因」。

## 2. 预注册读法（写在正式 T 场结果之前）

对象都是 T 场：训练 run 0–3，验证 run 4–5。种子 0。C、C_nolake 用已经选定的配置（η = 0.3；C 的 U = 3000，C_nolake 的 U = 2000）。C_skip 与 C 对齐（η = 0.3，U = 3000），不另选参。A 可选，用它自己的选定配置（η = 1，U = 3000），只作参照，不进下面的句子。

记 κ_lda(臂, 层) 为 9 名被试验证 Cohen's κ 的均值：收缩 LDA（`lsqr` + `shrinkage="auto"`）在训练试次的该层状态上拟合。试次状态是合格子窗活动的平均，和 `eval/scoring.py` 给出水口做的平均是同一种归约。κ_out(臂) 是出水口自己的冻结聚合（11 个子窗取平均再 argmax），不是 LDA。界仍用 0.05，和 M14 的用词同一档，但结论标成探索性。

| 编号 | 成立的条件 | 读法 |
| --- | --- | --- |
| D1 | κ_lda(C, 池) 与 κ_lda(C_nolake, 池) 相差 < 0.05，且 κ_lda(C, 池) − κ_lda(C, 微柱) ≥ 0.05 | 池里还有和 C_nolake 相当的信息，丢掉发生在池 → 微柱这一跳 |
| D2 | κ_lda(C, 微柱) 与 κ_lda(C, 池) 相差 < 0.05，且 κ_lda(C, 微柱) − κ_lda(C, 出水口) ≥ 0.05 | 微柱里还有，丢掉发生在微柱 → 出水口 |
| D3 | κ_lda(C, 出水口) − κ_out(C) ≥ 0.05 | 出水口状态里线性可读的信息多于奇读出实际用掉的 |
| D4 | κ_lda(C_nolake, 池) − κ_lda(C, 池) ≥ 0.05 | 池本身已经低于 C_nolake，损失在进沙盘之前（前端或 C 没把池训到同一水平），不是沙盘内部的一跳 |
| S1 | κ_out(C_nolake) − κ_out(C_skip) < 0.05，且 κ_out(C_skip) − κ_out(C) ≥ 0.05 | 补上池 ↔ 出水口之后缺口合上：缺的跳连是充分原因 |
| S2 | κ_out(C_skip) − κ_out(C) < 0.05，且 κ_out(C_nolake) − κ_out(C_skip) ≥ 0.05 | 跳连不是充分原因 |
| S3 | C_skip 相对 C 的增益 ≥ C 与 C_nolake 缺口的一半，但缺口仍 ≥ 0.05 | 跳连是原因的一部分 |

D1 与 D4 互斥。S1 与 S2 互斥。哪条都不满足就写成「这一档界线分不开」，不另换界。

每个准确率同时写合格率（子窗收敛比例）和拒判试次数。探针和 C_skip 的合格率若低于 0.98，先记收敛问题，不解释成结构结论。

### 2.1 已知的初始化差别（不是结果，是建图事实）

C 在初始化时，自由→自由的行质量上限卡在池上：池预算 0.3 加读入预算 0.5，正好等于 `RHO0_MAX = 0.8`，全局缩放是 1。微柱上的行质量大约 0.24–0.29，出水口大约 0.50。三条预算 0.5 + 0.15 + 0.5 并没有叠在同一个神经元上。

C_skip 的 `skip` 预算是 0.5，和 `out` 同一档。抽权重在 C 的全部随机数之后，所以缩放前的共有边和 C 相同。缩放后，出水口同时吃到 `out` 和 `skip`，行质量在全局缩放前大约 0.93–1.00（种子 0/1，被试 1、2、9），全局缩放大约 0.80–0.86，并且乘在**所有**自由边上，包括与 C 共有的池、微柱和出水口。因此 C_skip 不是「只多了一批突触、其余权重逐位相同」。`build()["init_scale"]` 把每条通路的缩放和这个全局因子记下来，回执的 `config.init_scale` 同样有。S2 若成立，不能单独说成「这条边没有用」，还要看这个 0.80–0.86 的整体缩小。

## 3. 实现

- `ep_eeg/m14.py`：臂 `C_skip`。池 ↔ 出水口互惠、可塑，通路名 `skip`。探索性臂拒绝 `split=gate`，回执默认写到 `receipts/M14/exploratory/`，不进入 `m14_report` 的预注册臂列表，也不进入 `selected.json`。
- `ep_eeg/m14_probe.py`：分层探针。有权重文件就加载，否则按选定配置重训并保存 `u<U>.npz`。重训时不做中途验证（验证不改参数）。输出 `probe.tsv` 和 `probe.md`。
- `ep_eeg/m14_report.py --exploratory`：只汇总 `split=val` 的探索性回执。

## 4. 正式运行（只碰 T 场）

权重不在仓库里（`receipts/M14/weights/` 被忽略）。探针发现没有 `u<U>.npz` 就会重训。按 M14 的计时，C 在 U = 3000 时每人大约 20 分钟，C 与 C_nolake 九人大约数小时；A 再加一份。不要加 `EPEEG_GATE`。

```bash
export MNE_DATA=/workspace/mne_data
export NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd /workspace

# 探针。C、C_nolake 用选定的 η=0.3 和各自的 U；A 用选定的 η=1、U=3000。种子 0。
python -m ep_eeg.m14_probe --arms C C_nolake A --subjects 1 2 3 4 5 6 7 8 9 --seed 0
# 输出 receipts/M14/diag/probe/probe.tsv 与 probe.md

# 只要 C 和 C_nolake 时可以去掉 A。

# C_skip：与 C 同一组 (η, U, 种子)，T 验证，回执在 receipts/M14/exploratory/
for s in 1 2 3 4 5 6 7 8 9; do
  python -m ep_eeg.m14 --subject $s --arm C_skip --seed 0 --U 3000 --eta 0.3 --split val --eval-every 250
done
python -m ep_eeg.m14_report --exploratory

# 训练之后也可以对 C_skip 做同一套探针（会读上面写下的权重）：
python -m ep_eeg.m14_probe --arms C_skip --subjects 1 2 3 4 5 6 7 8 9 --seed 0 --out receipts/M14/diag/probe_C_skip
```

开发冒烟（合成片段，`dev=true`，不加载任何会话）：

```bash
python -m ep_eeg.m14_probe --dev --arms C C_nolake --U 1
# 输出 receipts/M14/diag/dev/
```

## 5. 本环境实际跑了什么

IV-2a 的 T 场文件不在这台机器上（没有 `/workspace/mne_data`），也没有 M14 的权重检查点。正式的九人探针和 C_skip 训练没有跑。单元测试和一次合成数据的开发探针已跑，见 §6。开发探针的 κ 不能用来判断 D1–D4 或 S1–S3。

## 6. 开发运行（`dev=true`，不进判定）

`python -m pytest ep_eeg/test_m14_diag.py -q`：11 项通过。覆盖池↔出水口的边、初始化收敛证书、共有边只差记录下来的全局缩放、C_skip 拒绝非 T 场的 split、预注册选参不收录 C_skip、探针源码没有留出集加载器、试次平均与冻结聚合一致，以及一次不调用数据加载器的合成探针。

`python -m ep_eeg.m14_probe --dev --arms C C_nolake --U 1` 写入 `receipts/M14/diag/dev/`（墙钟 1.7 s）。合格率 1，拒判 0。合成片段上收缩 LDA 的验证 κ 在池、微柱、出水口都是 1，出水口自身判决的 κ 是 0。这只说明两条读法是分开算的：植入的类条件节律在 8 个试次上线性可分，奇出水口在一次更新之后还停在机会水平。不能用来判断 D1–D4。
