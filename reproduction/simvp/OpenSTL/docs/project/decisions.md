# 关键技术决策记录

> 格式：每条决策编号 DEC-XXX，含日期、背景、决策内容、备选方案、原因、影响。

## DEC-001 使用 gSTA 作为 SimVP 默认骨干

### 日期
2026-09-29（登记）

### 背景
SimVP 的隐空间翻译支持 13 种 MetaFormer 骨干，需要为 reproduction 选定默认配置。

### 决策内容
默认采用 `model_type='gSTA'`（SimVP v2，`GASubBlock`），对应 `configs/mmnist/simvp/SimVP_gSTA.py`。

### 备选方案
IncepU（SimVP v1）、ConvNeXt、Swin、ViT 等。

### 原因
gSTA 是 OpenSTL 中 SimVP v2 的推荐默认，参数量/性能均衡，且与上游论文对齐。

### 影响
后续所有基线实验默认以 gSTA 为准；对比实验可切换其他骨干但不改变默认。

---

## DEC-002 直接多帧输出 vs 自回归必须与 CIRA-Diff 公平比较

### 日期
2026-09-29（登记）

### 背景
SimVP 默认一次性直接输出全部未来帧；CIRA-Diff 为自回归扩散模型。

### 决策内容
在将 SimVP 与 CIRA-Diff 对比时，必须明确多步预测协议（direct multi-step vs autoregressive），
并在同一 `aft_seq_length` 下比较。

### 备选方案
仅用 SimVP 默认 direct 输出与 CIRA-Diff 比较（不公平）。

### 原因
原始 reproduction `Known issues` 明确提示该公平性风险。

### 影响
`docs/research/hypotheses.md` 与实验设计须体现该协议约束。

---

## DEC-003 通过自定义 dataloaders 接入 GOES-13 数据，不修改 OpenSTL 上游注册代码

### 日期
2026-09-30

### 背景
需要把 EDM GOES-16 ABI ch13 的 zarr 数据接入 OpenSTL 训练流程。走标准路径
（`tools/train.py -d goes13`）要求同步改动至少 4 个上游文件：
`dataset_constant.py`、`datasets/__init__.py`、`load_data` 分发逻辑，以及 parser
中 `--dataname` 的 `choices`。这与「保留上游 provenance」相冲突。

### 决策内容
采用 OpenSTL 官方支持的 `BaseExperiment(args, dataloaders=...)` 注入方式
（与自定义数据教程同一机制），并**只新增文件**：

- `openstl/datasets/dataloader_goes13.py`（dataset adapter，遵循既有 `dataloader_*.py` 约定）
- `configs/goes13/simvp/SimVP_gSTA.py`（超参，遵循 `configs/<dataname>/simvp/` 约定）
- `tools/train_simvp_goes13_smoke.py`（可执行入口，遵循 `tools/` 约定）

### 备选方案
A) 注册为标准 dataset 名（改动 4 个上游文件）；B) 完全独立的训练脚本、绕开 OpenSTL
训练框架（失去与 SimVP/Lightning 流水线的一致性）。

### 原因
方案 A 破坏上游 provenance 且提高后续 upstream merge 成本；方案 B 无法达到「验证整个
训练流程」的目的。当前的注入方式零上游改动，却完整走通 dataloader→Lightning→指标的
标准链路。

### 影响
约定：若未来需要 `-d goes13` 命令行集成，应在 provenance 审计记录的基础上再做注册
决策，而非临时改动上游。三条已知偏差（D1–D3）已在脚本 docstring 中显式记录。
