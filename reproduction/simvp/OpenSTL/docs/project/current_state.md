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
- ⚠️ 上一条（48 行）已过时：当时用 **base python** 探测，误判为环境缺失。真实情况见下条。

### 2026-09-30（二）（修改人：AI assistant，依据 `docs/project/ai_prompt.md`）

修改原因：定位并解决阻塞级问题；同时记录用户删除参考脚本的影响。

影响范围：

| 变更 | 路径 |
| --- | --- |
| slurm：改用专用 env + 依赖预检扩充 | `test_dl/test_dl.slurm` |
| 环境配方（实测可用） | `docs/dev/environment.md`（重写） |
| 新增诊断条目 | `docs/dev/debugging.md`（DBG-005 依赖链、DBG-006 参考脚本删除） |
| reproduction gate 表更新 | `reproduction/simvp/README.md` |

当前状态更新：
- **阻塞已解除**：真正的缺口不是 timm/lightning/fvcore（这些在 conda env `test` 里本来就有），
  而是 `cv2` / `imageio` / `pywt` / `skimage` 缺失 + `timm 1.0.30` 移除了 Nadam。
  已新建 conda env `openstl`（克隆 `test` + `timm==0.9.12` + 四个缺包）解决。
- **首次端到端运行成功**（CPU 极小验证）：`--n-train 8 --n-val 4 --n-test 4 --epochs 1 --cpu`
  跑通 train / val / test，产出 `checkpoints/{best,last,best-epoch=00-val_loss=0.348}.ckpt`、
  `saved/{inputs,preds,trues,metrics}.npy`、`model_param.json`、`train_<时间戳>.log`，
  模型 4.3 M 参数。这验证了 adapter、配置、两个 prepare 到的路径与前文的产物推断都正确。
- `Environment Working` 已达成；正式 GPU 上的 `Training Reproduced` 仍未达成。
- **`test_dl/test_single_sample_Chase2025.py` 已被删除**：不影响运行（无代码 import），
  仅 4 处文档/注释引用失效；其配色与数据约定已固化进新脚本，详见 DBG-006。

下一步：
1. 用 slurm 跑 GPU 版 EXP-002 小样本，把 EXP-002 文档更新为 `EXECUTED`。
2. 跑 EXP-003 单样本并判读。
3. full-split 训练前务必 `--no-preload` 或改分块统计（全量 preload 约 28 GB）。

#### 补充验证（同日，conda env `openstl`，CPU）

- **EXP-003 单样本脚本实跑通过**（`--index 0 --epochs 5`）：loss 0.8855 → 0.4076；
  产出 `sample_0_{comparison,error,loss_curve}.png`、`sample_0_{condition,truth,pred,error}.npy`、
  `sample_0_summary.json`（mse 0.2102 / rmse 0.4585 / mae 0.3317 / r2 0.6109，
  **仅 5 epoch，不足以下任何结论**）；同时在 `work_dirs/<ex_name>/` 下写出 ckpt 与
  `model_param.json`，**没有** `saved/*.npy` —— 与"未调 `exp.test()`"的推断一致。
- **单元测试实跑通过**：`pytest tests/test_models/test_simvp.py -q` → **6 passed**
  （含 gSTA / IncepU / ConvNeXt / Swin 参数化用例，以及非法 `model_type` 的断言用例）。
- 至此阻塞级问题全部解除；未验证项只剩 **GPU（slurm）路径**。

### 2026-09-30（四）—— slurm GPU 跑通 + 悬空引用 / 运行期护栏
修改原因：用户已用 slurm 实跑（job 102，节点 test-ai，单卡 RTX 4090），据日志收尾。
影响范围：

| 变更 | 路径 |
| --- | --- |
| 删除指向已删脚本的悬空引用（DBG-006） | `dataloader_goes13.py`、`train_simvp_goes13_single_sample.py`（docstring ×2）、`docs/experiments/EXP-003.../README.md`、`docs/project/knowledge.md` |
| 新增 preload 内存护栏（DBG-007） | `tools/train_simvp_goes13_smoke.py` |
| 状态更新 | `docs/experiments/EXP-002.../README.md`、`reproduction/simvp/README.md`、`docs/dev/debugging.md`（DBG-007） |

当前状态更新：
- **GPU 路径已验证**：slurm job 102 在节点 test-ai、单卡 RTX 4090 跑通 3 epoch；
  val_loss 0.365→0.295，测试 mse 7322.57 / mae 14507.10 / rmse 85.57。
  脚本里 `#SBATCH --gres=gpu:1` 已保证独占一张卡，原担心的「8 卡抢 0 号」坑已规避。
- **这些数字不是复现结论**：仅 3 epoch / 192 样本 + 归一化取自子集，仅供流水线验证。
- 悬空引用已全部改指向 `EXP-001-cira-diff-data-audit.md` 或脚本内固定约定。
- 运行期护栏 DBG-007：当 preload=True 且总样本 > 4000 时打印 OOM 警告，提醒全量训练加 `--no-preload`。
- EXP-002 文档升级为 `EXECUTED`；reproduction gate 标记为 `Training Reproduced (smoke)`。
