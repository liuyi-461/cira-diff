# 实验记录索引

> 每个实验按 `EXP-XXX` 编号，记录条件与结果，禁止只保存最终指标。

## 模板（EXP-001）

```markdown
# EXP-001

日期：
实验目的：
数据：（数据集、切分、pre/aft seq length）
方法：（model_type、config 路径）
参数：（lr、batch_size、epoch、seed）
结果：（MAE、MSE、FLOPs、吞吐）
结论：
下一步：
复现教程: 实现项目地址及运行步骤文档地址
```

## 当前实验
- **EXP-004**（统一 baseline + SimVP 全量评估）：见 `docs/experiments/EXP-004-simvp-vs-baseline/`。
- **EXP-005**（多 seed rollout 方差）：见 `docs/experiments/EXP-005-simvp-multiseed/`。
- 两者结论均由脚本自动生成，勿手改。

---

# EXP-002 SimVP 小样本训练流程验证（GOES-16 ABI ch13）

日期：
2026-09-30

实验目的：
在投入完整训练之前，用小样本端到端验证 OpenSTL/SimVP 的训练流水线是否可用
（数据读取 → DataLoader → Lightning 训练循环 → 验证 → 测试指标 → 产物落盘）。
本实验**不产生任何科学结论或可引用的指标**。

数据：
- `edm_GOES_ch13_train_dataset.zarr`（EDM GOES-16 ABI Channel 13，IR 亮温）
- 样本结构：ordinary 2→1；`input_images (N,2,256,256)` → `output_images (N,1,256,256)`，float16
- 小样本子集：`sample_indices(..., seed=42)`，默认 train 128 / val 32 / test 32，随机不重复抽样
- 归一化：z-score。**统计量子集来自抽样出的 train 子集**，非全量 train split（偏差 D3）

方法：
- SimVP + gSTA（`model_type='gSTA'`），通过 `BaseExperiment(dataloaders=...)` 注入自定义 loader
- `pre_seq_length=2`、`aft_seq_length=1`、`in_shape=[2,1,256,256]`
- 损失 MSELoss；SimVP 在 `aft < pre` 时先预测 2 帧再截断前 1 帧

参数：
- 见 `configs/goes13/simvp/SimVP_gSTA.py`（smoke 版：`hid_S=32, hid_T=256, N_S=4, N_T=4`）
- `lr=1e-3`、`batch_size=8`、`sched='onecycle'`、`epochs=3`（CLI 默认）
- 指标：`['mse', 'mae', 'rmse']`

结果：
**未执行（NOT EXECUTED）**。仅完成静态校验：三份代码 `py_compile` 通过；数据集契约在真实
数据上抽样验证（x=(2,1,256,256)、y=(1,1,256,256)，mean≈0.518、std≈0.441，min/max
[-3.742, 1.924]，无 NaN）。环境缺少 `timm` / `lightning` / `fvcore`，训练循环尚未运行。

结论：
流水线代码与数据契约已就绪；瓶颈在环境依赖，而非实现。

下一步：
1. 安装 `timm`、`lightning`、`fvcore`、`opencv-python` 后执行下述命令，确认训练/测试各跑通一轮；
2. 再扩大样本量并回到全分辨率超参（`hid_S=64, hid_T=512, N_T=8`），产出可比对的指标证据包；
3. 指标入 `docs/experiments/EXP-002-simvp-goes13-smoke/`。

复现教程:
实现：`reproduction/simvp/OpenSTL/tools/train_simvp_goes13_smoke.py`；
适配器：`openstl/datasets/dataloader_goes13.py`；
运行步骤与证据包：`docs/experiments/EXP-002-simvp-goes13-smoke/`。

---

# EXP-003 SimVP 单样本过拟合 + 预报可视化（GOES-16 ABI ch13）

日期：
2026-09-30

实验目的：
验证**模型与结构本身**是否正确（EXP-002 只验证流水线能否跑通）。取向单个样本训练至
过拟合：结构接线正确的 SimVP 应能把单样本 loss 压到极低并复现目标帧；通道错位、时间轴
错置、skip 连接断裂等问题会直接在并列图上暴露。

数据：
- 同一份 `edm_GOES_ch13_train_dataset.zarr`，**单个样本** `--index 0`（默认）
- 归一化统计量取自参考池 `--ref-samples 32`（偏差 D4：不取单样本自身，否则 std 退化）

方法：
- SimVP + gSTA；`pre_seq_length=2` / `aft_seq_length=1` / `in_shape=[2,1,256,256]`
- 单样本 loader：`batch_size=1`、`shuffle=False`、`drop_last=False`（单样本不可丢弃）
- 训练后 `exp.method` 前向推理，反归一化后出图

参数：
- `epochs=100`、`lr=1e-3`、`batch_size=1`（每 epoch 1 step）
- 由 `configs/goes13/simvp/SimVP_gSTA.py` 提供网络超参
- 色标沿用既有脚本：`Spectral_r`，`vmin=-4, vmax=2`

结果：
**未执行（NOT EXECUTED）**。仅完成静态校验（`py_compile` 通过、关键符号引用核对、
清理未使用导入）。环境缺 `timm` / `lightning` / `fvcore`。
判读准则见证据包 `docs/experiments/EXP-003-simvp-goes13-single-sample/`。

结论：
验证用的脚本与判据已就绪；尚未获得 loss 曲线与可视化证据。

下一步：
1. 安装依赖后执行，记录 loss 首尾值与 MAE/RMSE，更新为 `EXECUTED`；
2. 换 2–3 个 `--index` 复核；
3. 通过后再执行 EXP-002 多样本流水线与 full-split 训练。

复现教程:
实现：`reproduction/simvp/OpenSTL/tools/train_simvp_goes13_single_sample.py`；
适配器：`openstl/datasets/dataloader_goes13.py`；
配置：`configs/goes13/simvp/SimVP_gSTA.py`；
运行步骤与判读准则：`docs/experiments/EXP-003-simvp-goes13-single-sample/`。

---
<!-- INDEX:BEGIN -->
# EXP-004 统一 Baseline 协议 + SimVP 全量训练评估（GOES-16 ABI ch13）

日期：
2026-10-07

实验目的：
把外部基线（ly 的 Vanilla UNet、cb 的 SimVP）固化为可引用 baseline，定义统一评估协议，并在同一把尺子上评估我们的 SimVP。

数据：
- train/validation/test **三份独立 zarr**（test = 官方 split，1024 条）
- `output_images` (1024,18,256,256)；单步任务取第 0 帧 = **t+10min**（已实测 |Y[:,0]-X[:,1]| 与已知 10 分钟步长比值 1.003）
- zarr 内已是归一化空间（mean=0, std=1）；`Tb(K) = zarr × 19.3297 + 279.0699`

方法：
- SimVP + gSTA，4.705M 参数，通过 `BaseExperiment(dataloaders=...)` 注入
- `pre_seq_length=2` / `aft_seq_length=1` / `in_shape=[2,1,256,256]`
- **不做子集 z-score**（修正 EXP-002 的 D3 偏差），模型直接吃 zarr 原值
- 评估用 `tools/eval_simvp_baseline.py`（加载 best.ckpt、逐像素口径）

参数：
- `configs/satcast/SimVP.py`：`hid_S=64, hid_T=256, N_S=2, N_T=4, drop_path=0.1`
- `lr=2e-3`、`sched=onecycle`、`batch_size=8`、`epoch=100`、`val_batch_size=32`
- seed=42（多 seed 见 EXP-005）

结果：
单步（test 全量 1024 条，best epoch = 30）：
- SimVP：MSE=0.007336、**RMSE=1.6556 K**、MAE=0.8698 K、SSIM=0.8980
- persistence：MSE=0.055390、RMSE=4.5492 K
- 对照：UNet(ly,seed42) RMSE≈1.74 K；SimVP(cb,100ep) RMSE≈1.57 K
- MSE 相对 persistence 降低 **86.76%**
18 步 rollout（128 条子集）：LT=1 RMSE=1.54 K → LT=18 RMSE=17.10 K
- **从 LT=10（100 分钟）起 RMSE 反超 persistence**
- LT=18 对照 UNet(seed0) MSE 0.419200 vs 本模型 0.782793

结论：
1. **单步：SimVP 优于 Vanilla UNet**（MSE 好 9.3%），且远优于 persistence。
2. **rollout：SimVP 长时稳定性明显弱于 UNet**，且在中长时效（约 100 分钟后）会劣于 persistence —— 纯单步 teacher-forced 训练的误差累积所致。
3. 上述 rollout 结论为**单 seed**，稳健性由 **EXP-005** 判定。

下一步：
1. 用 EXP-005 的多 seed 方差确认 rollout 结论是否稳健；
2. 尝试 rollout-aware 训练（scheduled sampling / multi-step loss）改善长时效；
3. 与 CIRA-Diff 在统一多步协议下对比（DEC-002）。

复现教程:
实现：`tools/train_simvp_satcast.py` + `tools/eval_simvp_baseline.py`；
适配器：`openstl/datasets/dataloader_satcast.py`；
配置：`configs/satcast/SimVP.py`；
提交：`test_dl/train_baseline.slurm` / `test_dl/eval_baseline.slurm`；
证据包：`docs/experiments/EXP-004-simvp-vs-baseline/`。

---

# EXP-005 SimVP 多 seed rollout 方差（与 UNet 对照）

日期：
2026-10-07

实验目的：
ly 证明 Vanilla UNet 的 rollout 对 seed 极敏感（LT=18 CV=13.4%），单 seed 之间比 rollout 无意义。本实验训练 SimVP 多个 seed，拿到 SimVP 自身的 rollout 方差，判定 EXP-004 中"LT=18 比 UNet 差 ~60%"是否稳健。

数据/方法/参数：
- 与 EXP-004 **完全相同**的配置与数据，**仅改变模型初始化 seed**
- seed 集合：0 / 42 / 123（与 ly 一致）
- 每 seed 评估单步（1024 条）+ 18 步 rollout（128 条子集）

结果：
**多 seed 结果尚未产出**（等 seed 0/123 训练完成后由 `aggregate_multiseed.py` 汇总）。

结论：
待产出。

下一步：
1. 若结论稳健 → 针对长时效做 rollout-aware 训练；
2. 若增加 seed 数（≥5）以降低均值标准误（≈CV/√n）；
3. 结论回填 EXP-004 并更新 `docs/research/conclusions.md`。

复现教程:
训练：`SEED=0 sbatch test_dl/train_baseline.slurm`（同理 123）；
汇总：`tools/aggregate_multiseed.py` → `docs/experiments/EXP-005-simvp-multiseed/`。
<!-- INDEX:END -->
