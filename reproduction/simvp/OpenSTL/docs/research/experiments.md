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
- 暂无（reproduction 仍处 candidate only，见 `docs/project/current_state.md`）。

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
