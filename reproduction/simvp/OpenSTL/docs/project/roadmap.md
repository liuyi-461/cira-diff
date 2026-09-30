# Roadmap

## Phase 1 — Surveyed / Code Available
目标：完成上游审计与环境就绪。
任务：
- 审计 SimVP 论文与 OpenSTL 仓库关系（provenance）。
- 按 `environment.yml` / `requirements.txt` 建立 conda 环境。
- 下载或确认 OpenSTL 代码完整性。

## Phase 2 — Downloaded / Environment Working
目标：可运行训练 / 测试流水线。
任务：
- 选定数据集，编写数据 adapter（见 `docs/data/`）。
- 修复过期单元测试（见 `docs/dev/debugging.md`）。
- 在目标数据集上跑通 `tools/train.py` 与 `tools/test.py`。

## Phase 3 — Inference / Training / Evaluation Reproduced
目标：产出可对比的指标证据包。
任务：
- 复现 SimVP 基线指标（MAE / MSE）。
- 在 `docs/experiments/` 沉淀单个实验证据包。
- 与 CIRA-Diff 在统一多步协议下对比（见 `docs/project/decisions.md` DEC-002）。

## Phase 4 — 可复用包
目标：将验证过的流程沉淀为 `src/satforecast/`（Lab 级，超出本 OpenSTL 目录范围）。
