# 当前版本与状态

## 已完成
- OpenSTL 代码已 clone 至 `reproduction/simvp/OpenSTL/`。
- 完成对 SimVP 实现的核心代码理解：`simvp_model.py`、`simvp_modules.py`、`methods/simvp.py`、
  `api/exp.py`、`tools/train.py`。
- 建立本文档体系（见 `docs/`）。

## 当前实现
- `SimVP_Model`：Encoder(`ConvSC`) + `MidMetaNet`(gSTA 默认) + Decoder(`ConvSC` + skip)。
- `SimVP` method：MSELoss 监督，支持直接多帧预测与递归自回归。
- 配置：`configs/<dataname>/simvp/SimVP_gSTA.py` 等 13 种骨干变体。

## 当前问题
- reproduction 状态仍为 `candidate only`：论文 / 官方仓库关系未审计，数据 / 权重 / 环境未就绪。
- `tests/test_models/test_simvp.py` 已过期，调用签名与当前 `SimVP_Model` 不一致，会失败。
- `README.md`（reproduction 级）中 Paper / Dataset / Weights / Environment 等均为 `TODO`。

## 下一步计划
- 审计 SimVP 与 OpenSTL 的上游 provenance 关系（见 `docs/reproduction/`）。
- 选定目标数据集并编写数据 adapter（见 `docs/data/`）。
- 修复 / 补充测试（见 `docs/dev/`）。
- 跑通 environment → inference → training → evaluation，逐状态迁移并留档。

---

## 更新记录

### 2026-09-30（修改人：AI assistant，依据 `docs/project/ai_prompt.md`）

修改原因：完成 provenance 审计并接入 GOES-13 数据集，产出可跑通整个训练流程的小样本脚本。

影响范围（均为**新增文件**，未改动任何上游/既有文件内容）：

| 变更 | 路径 |
| --- | --- |
| Provenance 审计 | `docs/reproduction/provenance_audit.md` |
| Dataset adapter | `openstl/datasets/dataloader_goes13.py` |
| 超参配置 | `configs/goes13/simvp/SimVP_gSTA.py` |
| 训练入口（多样本流水线） | `tools/train_simvp_goes13_smoke.py` |
| 训练入口（单样本 + 可视化） | `tools/train_simvp_goes13_single_sample.py` |
| 测试修复 | `tests/test_models/test_simvp.py`（旧用例已失效，重写） |

当前状态更新：
- 数据集已选定并完成契约验证：EDM GOES-16 ABI ch13，ordinary 2→1 帧，256×256，float16。
- Dataset adapter 已完成；两个训练脚本（EXP-002 多样本流水线、EXP-003 单样本 + 可视化）
  均已完成并通过语法编译与符号核对。
- **尚未执行训练**：环境缺 `timm` / `lightning` / `fvcore`，故 Reproduction 状态仍跳过
  `Environment Working`，不得声称 `Training Reproduced`。
- 建议执行顺序：EXP-003（单样本，先确认模型/结构正确）→ EXP-002（多样本，确认流水线）
  → full-split 训练。
