# 当前版本与状态

## 已完成
- **SimVP 2→1 全量训练复现成功**：100 epoch，test 1024 条单步 RMSE **1.5761 K**，
  与 cb 独立复现的 1.5745 K 仅差 0.1% → 交叉验证。
- **baseline 协议闭环**：三份独立 zarr、物理常数归一化（不做子集 z-score）、
  逐像素口径、亮温 K；persistence 全量独立复算与 cb 偏差 **0.0000%**。
- **多 seed + 配对检验**：3 seed（0/42/123），LT=1..18 **全部显著优于 UNet**
  （配对 t 达 −11 ~ −34.7），推翻早期基于聚合 std 的"不显著"假阴性。
- **抗模糊方案对照**（EXP-006，七臂）：GAN 失败（LT=18 恶化 34.8%、锐度比冲到 2.97 成噪声）；
  多步 + scheduled sampling **有效**（3 seed：LT=18 改善 **15.1% ± 1.0 pp**，12.3σ），
  但**锐度同时降到 baseline 的 49%**。
- **感知-失真权衡实测**（EXP-007）：SimVP / UNet / EDM 同批 342 条样本配对——
  **MSE 最优的模型不是最清晰的**（EDM 锐度 0.902 vs SimVP 0.822，但 MSE 差 49%）。
- **主线报告**：`docs/research/forecast-realism-report.md`。

## 当前实现
- 模型：SimVP + gSTA，**4.705M** 参数，**2 帧 → 1 帧**（帧间隔 10 min）；
  长时效靠自回归外推，已评估到 **18 步 = 3 小时**。
- **自研代码入口：`tools/README.md`**（训练/评估/对比/报告脚本，含完整 CLI、实测结论、9 条踩坑记录）。
- 数据适配器：`openstl/datasets/dataloader_satcast.py`（上游唯一新增文件）。
- 超参唯一来源：`configs/satcast/SimVP.py`。
- 提交脚本：`test_dl/*.slurm`（环境自检 + GPU 独占 + 依赖链自动串联）。
- 预测数组已导出为官方格式 `(1024,18,256,256)`：`work_dirs/satcast_preds/`（11 个文件）。

## 当前问题
- **"清晰"与"低 MSE"在本框架内不可兼得**（已有四条独立证据）；若目标是预报真实性，
  应转向生成式路线（EDM/扩散 + 集合 + 感知类指标），而非继续在确定性框架内调优。
- EDM 的 LT≥2 数据受 `ly` 的 rollout Bug 1 污染，长时效锐度对比尚缺；
  且 EDM 为 113.7M / 825 ep，与 SimVP（4.7M / 100 ep）**规模不对等**，引用须注明。
- scheduled sampling 的 15.1% 收益基于 1024 条多步数据（baseline 用 35595 条），
  数据规模差 35 倍，归因时需注意。
- `tools/train.py`、`tools/test.py` 为上游残留，与本项目流程无关。

## 下一步计划
1. 把 SS 臂与 baseline 的 **锐度–MSE 散点图**补进 `docs/figs/`（目前只有数字表）。
2. 补 EDM 长时效锐度（需 `ly` 用修复后 rollout 重跑，只读协作）。
3. 若推进生成式路线：在 SimVP 基线上引入 EDM/扩散，配套集合与 CRPS / spread-skill 指标。
4. 若可获得更多多步真值，用更大数据量复核 SS 的 15.1% 收益。

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

### 2026-10-14（修改人：AI assistant）—— 全量复现 + 七臂对照 + 感知-失真权衡主线
修改原因：项目已从"流水线冒烟"推进到"可对外引用的横向结论"，状态文档需同步；
同时为 `tools/README.md` 建立文档入口。

影响范围：

| 变更 | 路径 |
| --- | --- |
| 新增自研代码文档（含 CLI / 实测结论 / 9 条踩坑记录） | `tools/README.md` |
| 文档索引增加「代码入口」表 + 推荐阅读顺序 | `docs/README.md` |
| 状态文档四个小节刷新 + 本更新记录 | `docs/project/current_state.md` |
| 主线结论材料 | `docs/research/forecast-realism-report.md` |
| 新增实验证据包 | `docs/experiments/EXP-004~007/` |
| 图表集扩充至 10 张 | `docs/figs/` |

当前状态更新（均已在 test 集 1024 条上按统一协议验证）：

- **复现成功**：SimVP 单步 RMSE **1.5761 K**（cb 独立复现 1.5745 K，差 0.1%）；
  persistence 全量独立复算与 cb 偏差 **0.0000%** → 口径链路闭环。
- **显著优于 UNet**：3 seed 配对检验，LT=1..18 全部显著（t 达 −11 ~ −34.7）。
- **抗模糊尝试的结论是负向但有价值**：GAN 使 LT=18 恶化 34.8% 且锐度比冲到 2.97（噪声）；
  唯一有效的是 scheduled sampling（3 seed 改善 **15.1% ± 1.0 pp**，12.3σ），
  但其机制是**让预报更平滑**（锐度 0.273 → 0.133）。
- **核心结论（主线）**：确定性模型 MSE 更优但必然模糊；生成式 EDM 更清晰（0.902 vs 0.822）
  但 MSE 差 49%。**感知-失真权衡已被四条独立证据证实**，正反方向都跑通。
- 上游 `openstl/` 除新增 `datasets/dataloader_satcast.py` 外**零修改**（DEC-003）。
