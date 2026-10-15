# EP-EEG：平衡传播动力系统解码运动想象 EEG

本项目用 [cadence](C:\project\cadence) 的平衡传播（EP）连续动力系统，解码 BCI Competition IV-2a 的四类运动想象 EEG。研究问题是：**零结构先验的全对称介质，经局部 EP 学习后，能否在解码性能和可解释拓扑上优于人工脑区先验和前馈基线？**

这是一个**自动化科研**项目。人负责设定目标、冻结边界和审阅阶段门；agent 在阶段内自主提出假设、做实验、记台账、决定保留或丢弃改动，不需要逐次请示。

## 资料

- `docs/plan.md`：研究方案原文。方案中的数值是初始假设；被 ADR 修订的部分以 ADR 为准。
- `docs/adr/`：已接受的设计决策及其理由。改动相关结构之前，先读对应的 ADR。agent 可以新写 `status: proposed` 的 ADR，由人在阶段门接受；推翻已有决策时，新写一条 ADR 取代它。
- `GLOSSARY.md`：项目术语。代码、报告和 ADR 统一使用其中的规范词。
- `docs/discussions/`：方案评审的原始讨论，共三轮：argon → Fable → argon。结论已收入 ADR。其中的示例代码针对 cadence 0.72.1 编写，有已知错误（见下文），需要按下表改写后才能使用。
- `C:\project\cadence`（cadence-net 0.83.0，commit `6499e56`）：底层库，**只读**。所需能力在本仓库的适配层实现；cadence 的缺陷或缺失的能力，记入阶段报告的「待决问题」。该仓库有 `.codegraph/` 索引，可用 `codegraph_explore`（projectPath 设为该目录）查代码。按需阅读：
  - 精确签名和默认值：`docs/api.md`
  - 能量函数与学习规则：`docs/learning.md`
  - 收敛证书：`docs/certificate.md`
  - 求解契约：`docs/contracts.md`
  - 自定义建图：`docs/connectomes.md`、`docs/build.md`
  - 后端：`docs/backends.md`
  - 证据规则：`docs/protocols.md`、`docs/receipts.md`

## cadence 事实（主线路径）

主线按 [ADR-0003](docs/adr/0003-neuralgraph-not-compose.md) 的决定，直接使用 `Genome`/`Connectome`、`NeuralGraph` 和 `Learner`。下表已于 2026-10-15 对照 cadence `6499e56` 核实；cadence 更新后需要重新核实并更新本表。

| 主题 | cadence 实际 | 本项目做法 |
| --- | --- | --- |
| 建图 | `Region(name, size)` 的名字不能含 `/`。`Projection(pre, post, density=1.0, sign=0.0, scale=1.0, count=1.0, reciprocal=True)` 中，sign 是平均符号（0.0 表示正负各半），scale 乘在 Glorot 幅值上。需要逐突触控制时用 `Connectome.from_synapses`，它的 `min_count` 只在建图时过滤突触。 | 输入投射单向；微柱之间、微柱与出水口之间双向（ADR-0002） |
| 神经元与能量 | `learning_neuron_model()` 的默认 dt=0.5、L=0.5。ρ 在 v>0 时为 tanh(v/2)，v<0 时为 0.1·tanh(v/2)，值域为 (−0.1, 1)。单向输入在能量函数中属于驱动项。 | 连续目标必须落在 (−0.1, 1) 内 |
| 收敛证书 | `cadence.certificate.certificate(graph)`：`.certified` 要求 rate<1 且无 adaptation，其中 rate = 1 − dt(1 − L·row_mass)，`mass_limit` = 1/L。row_mass 取各神经元入射有效权重绝对值之和的最大值，单向投射也计入。证书只覆盖自由相，学习后会变化。 | 初始化时和每个 epoch 结束时都检查（ADR-0001） |
| 学习 | `LearnerConfig` 不设参数时的默认值：beta=0.1、eta=0.2、free_steps=100、nudged_steps=50、tolerance=1e-4、decay=0、momentum=0、qualified=False；scale_cap=8，只裁剪单个突触的效能，不限制行和。`Learner.step(drive, labels)` 的 drive 形状必须是 (batch, conn.n)。 | config 每个字段都显式写出，不依赖默认值 |
| 合格学习 | `qualified=True` 时，微扰相只有 `nudged_steps` 的预算，未达到容限就抛出 `LearningPhaseError`。`nudged_steps < free_steps` 会触发 RuntimeWarning。 | nudged_steps 设得与 free_steps 相当 |
| β 与对称性保护 | 默认使用中心化的 ±β 对比。`contrast_asymmetry` 和自动折半 β 的机制只存在于 `TemporalPatchNet` 中。 | β=0.04 只是假设；需要对称性保护时，在适配层调用 `cadence.temporal.contrast_asymmetry` 自行实现 |
| 衰减与剪枝 | 只有乘性的 `decay`，没有 L1，也没有在线剪枝。 | L1 和剪枝在适配层实现，并作为实验自变量 |
| 拓扑打乱 | `cd.shuffled(connectome, seed, keep=None)` 只置换突触终点，不保留反向配对。 | 保持对称的打乱在本仓库实现（ADR-0006） |
| 出水口 | 低层接口默认没有侧抑制。多槽用 `Learner(slots=...)`，各槽大小之和等于输出数。`Bins` 是 9 档离散编码。连续目标用 `nudge="quadratic"`。 | — |
| 输入 | cadence 不压缩输入。 | 预处理层做 tanh(x/3) 压缩 |
| 权重读取 | `NeuralGraph.dense()` 返回 W[pre, post]。 | 用于拓扑对账 |

讨论记录中示例代码的已知错误：

- `Region(f"in/{c}")` 的名字不合法。
- 讨论 02 的 `qualified=True, nudged_steps=12` 会导致每次都拒学。
- 讨论 02 把输入投射设为双向，输入神经元因此变成了自由变量。
- 讨论 03 中 `nudged_steps=256` 小于 `free_steps=1024`，会触发 RuntimeWarning。

## 评估协议（冻结）

- **数据**：BCI IV-2a，通过 MOABB 的 `BNCI2014_001` 加载。共 9 名被试，每人有 T、E 两个会话，每个会话 6 个 run、288 个试次（四类各 72 个）。数据包含 22 导 EEG 和 3 导 EOG，采样率 250 Hz。预处理和输入编码按 [ADR-0005](docs/adr/0005-per-channel-preprocessing-no-ea.md) 和 [ADR-0004](docs/adr/0004-subwindow-dual-track-encoding.md) 执行。
- **被试内方案**：每名被试独立训练和评估。
- **验证集**：T 会话按时间顺序切分，run 0–3 用于训练，run 4–5 作为验证集。所有选参和保留/丢弃的决定都只看验证集。
- **搜索指标**：9 名被试验证 Cohen's κ 的均值；同时记录准确率、合格率和耗时。
- **留出集**：E 会话。研究循环中不加载 E。只在阶段门对该阶段冻结的每个候选评估一次，结果写入 `holdout.tsv`。看过 E 之后再修改方法，就要开启新阶段，并把旧的 E 结果标注为「已用于决策」。讨论记录里把 A01 的 E 会话当调参靶标，本项目不采用这种做法。
- **冻结评估层**：`eval/`，包括数据加载、切分和评分函数，在 M0 建立，之后研究循环只读取它。修改它需要开启新阶段，并在报告中写明理由。
- **对照公平**：组 1–3 只在方案规定的结构自变量上有差别，主结论只用 Track-2 特征。调参预算按台账中每组的实验条数计算；比较时各组预算应相当，否则如实报告差距。组 4 包括 FBCSP、EEGNet（用 MOABB 和 braindecode 的标准实现），以及与沙盘使用相同特征的**同输入 LDA**；同样只用 T 会话选参。
- **统计**：每个配置至少跑 3 个种子。组间比较以被试为单位做配对检验（Wilcoxon 符号秩），报告每名被试的数值和效应量。

## 研究循环

阶段内重复以下步骤，不需要停下来请示：

1. **恢复现场**：读取 `results.tsv` 末尾、`git log --oneline -20`、`git status` 和当前阶段的 `reports/M<n>.md`，与未提交的改动对账。
2. **写下假设**：用一句话写明改什么、预期指标往哪个方向变、依据是什么（方案、ADR 或台账中的某一行）。每次只改一个自变量。
3. **改动并提交**：修改代码，然后 `git commit`。
4. **运行**：按当前阶段的预算档位运行，输出写入 `runs/<commit>.log`。耗时超过档位上限的 2 倍就终止，记为 crash。crash 后先读日志末尾尝试修复；试几次仍不行就记账，换下一个想法。
5. **先记账，再取舍**：向 `results.tsv` 追加一行，列为 `commit  stage  arm  kappa_val  acc_val  settle_rate  seeds  wall_s  status  description`，status 取 keep、discard 或 crash。
6. **取舍**：如果验证 κ 的提升超过种子间的标准差，或者指标持平但代码更少，就保留（keep）；否则用 `git reset --hard` 回到上一个 keep。

**台账**指 `results.tsv`，它不纳入 git 跟踪，以免被 reset 清掉。预算档位分两档：**快速档**用于筛选；**完整档**是 9 名被试 × 3 个种子，用于确认是否保留。两档的具体数值在 M0/M1 实测后写入「环境」一节。

## 阶段与阶段门

每个阶段结束时写一份 `reports/M<n>.md`，内容包括：台账区间、最佳候选、留出集结果（每个候选只评一次）、负结果、待决问题和下一阶段计划。

- **M0 基础设施与基线**：执行 `git init`，搭建环境，建立冻结评估层、预处理和双轨特征，开始记台账；跑 FBCSP、EEGNet 和同输入 LDA。
  - **完成标准**：三个基线的完整档结果已记入台账；E 会话已对它们各评一次；打乱标签的对照处于机会水平（κ≈0）；已确认 EOG 标定段是否可用；「环境」一节已改为实测过的命令。
- **M1 cadence 可行性**：按 ADR-0001 至 ADR-0003 搭建沙盘，先在 A01 上跑通，再扩展到 9 名被试。
  - **完成标准**：初始化时和每个 epoch 的证书都通过；自由相和微扰相的合格率 ≥98%；验证 κ 不低于同输入 LDA，且显著高于打乱标签的对照；子窗聚合方式和「正常脑电段」的定义已有成文的决定。
- **M2 组 1–3 对照**：在调参预算相当的前提下完成三组实验。
  - **完成标准**：组 1–4 在同一协议下都有完整档结果和配对检验；E 会话对每组各评一次。
- **M3 拓扑对账**：按 [ADR-0006](docs/adr/0006-preregistered-topology-predictions.md) 检验预注册的预测。
  - **完成标准**：每条预测都给出成立或不成立的判定，并附有多种子结果和对照结果。
- **M4 坏导联补全**：测试时把 2–4 个导联（如 C3、C4）置零，所有组使用相同的断导集合。
  - **完成标准**：每组都有 κ 随断导数变化的退化曲线。
- **M5 出水口扩展**：冻结 M2 中组 1 的沙盘，只更换出水口（多槽或连续向量）。
  - **完成标准**：连续目标的来源已经由人确认，并有与基线的对比。

## 声明规则

- 残差小只说明数值收敛，不说明答案正确。报告每个准确率时，都要同时给出合格率和拒行数。
- 成本统计要包含全部求解，即自由相、微扰相和被拒的求解，并且各组使用同一套计时口径。
- 预注册以外的拓扑发现一律标为探索性。
- 结论用可测量的量来表述；方案中的比喻只用于解释。
- 负结果、crash 和被丢弃的尝试都保留在台账中。

## 待决问题

由 agent 在 M1 中决定并写入报告的：

- 一个试次的判决如何由 11 个子窗聚合得出。
- 「正常脑电段」的定义，即合格率 ≥98% 的分母。

交给人决定的：

- 是否做跨被试预训练（讨论 02 建议用其余 8 名被试预训练，以缓解约 4.1k 参数对 288 个试次的比例问题），以及预训练时如何避开各被试的 E 会话。
- M5 连续目标的来源：IV-2a 没有连续轨迹标签。

遇到以下情况，写入当前报告的「待决问题」，并继续做不受阻塞的工作：

- 需要在阶段门以外读取 E 会话
- 需要修改冻结评估层或各组的定义
- 需要修改 cadence 源码
- 需要引入新的数据集
- 需要接受或推翻 ADR

## 环境

- 运行平台：Windows + PowerShell。
- Python：系统自带 3.14，超出 cadence 已测试的版本范围（3.11–3.13），所以用 uv 建一个 3.12 的虚拟环境。已实测 Python 3.12.13。
- 已安装并导入验证：cadence-net 0.83.0（editable，commit `6499e56`）、MOABB 1.7.2、Braindecode 1.8.1、scikit-learn 1.9.1。
- 安装命令：
  ```powershell
  uv venv --python 3.12 .venv
  uv pip install --python .venv\Scripts\python.exe -e "C:\project\cadence[fast]" moabb braindecode scikit-learn
  ```
- MOABB 1.7.2 的 `BNCI2014_001.get_data(sessions=...)` 会在过滤前加载该被试的 T 和 E。禁止用它执行研究循环；M0 先实现只加载 T 的数据适配层。
- 硬件：本机没有 NVIDIA GPU。cadence 用 CPU 后端加 `[fast]`（numba）；据 cadence `docs/backends.md`，小规模网络在 CPU + Numba 上比 CUDA 更快。
- 初始命令如下，M0 验证后改为实测过的命令：
  ```powershell
  uv venv --python 3.12 .venv
  uv pip install --python .venv\Scripts\python.exe -e "C:\project\cadence[fast]" moabb braindecode scikit-learn
  ```
- 本节由 agent 维护，记录实测命令、预算档位和单被试耗时；对照表在重新核实 cadence 后也可以由 agent 更新。其余各节由人维护。
