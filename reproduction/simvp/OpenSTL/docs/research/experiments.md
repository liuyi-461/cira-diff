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
2026-10-09

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
单步（test 全量 1024 条，best epoch = 92）：
- SimVP：MSE=0.006648、**RMSE=1.5761 K**、MAE=0.7926 K、SSIM=0.9066
- persistence：MSE=0.055390、RMSE=4.5492 K
- 对照：UNet(ly,seed42) RMSE≈1.74 K；SimVP(cb,100ep) RMSE≈1.57 K
- MSE 相对 persistence 降低 **88.00%**
18 步 rollout（128 条子集）：LT=1 RMSE=1.47 K → LT=18 RMSE=11.02 K
- LT=18 对照 UNet(seed0) MSE 0.419200 vs 本模型 0.325102

结论：
1. **单步**：SimVP MSE 0.006648 vs UNet 0.008092 → SimVP 更好 **17.8%**；两者都远优于 persistence（4.5492 K）。
2. **rollout**：LT=18 SimVP 0.3251 vs UNet(seed0) 0.4192 → SimVP 更好 **22.4%**；**全程优于 persistence**。
3. rollout 的 **seed 方差**判定见 **EXP-005**（3 seed：0/42/123）。

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
2026-10-09

实验目的：
ly 证明 Vanilla UNet 的 rollout 对 seed 极敏感（LT=18 CV=13.4%），单 seed 之间比 rollout 无意义。本实验训练 SimVP 多个 seed，拿到 SimVP 自身的 rollout 方差，判定 EXP-004 中"LT=18 比 UNet 差 ~60%"是否稳健。

数据/方法/参数：
- 与 EXP-004 **完全相同**的配置与数据，**仅改变模型初始化 seed**
- seed 集合：0 / 42 / 123（与 ly 一致）
- 每 seed 评估单步（1024 条）+ 18 步 rollout（128 条子集）

结果：
单步（3 seeds）：SimVP MSE=0.006632±0.000018 vs UNet 0.008100±0.000054
rollout LT=18（3 seeds）：SimVP MSE=0.3860±0.0581（CV=15.1%）vs UNet 0.4884±0.0789（CV=16.2%）
- **SimVP 更低（更好），相对优势 21.0%**；差值 0.1024 vs 两侧 std 之和 0.1370
- 逐 LT 的 SimVP/UNet mean±std 与 CV 见 `docs/experiments/EXP-005-simvp-multiseed/`。

结论：
LT=18 的差异（0.1024）**未超过**两侧标准差之和（0.1370）→ n=3 下**统计不显著**：SimVP 呈更优趋势，但不能作因果性断言。
SimVP 自身 rollout CV（LT=18）=15.1%，可与 UNet 的 13.4% 直接对照。

下一步：
1. 若结论稳健 → 针对长时效做 rollout-aware 训练；
2. 若增加 seed 数（≥5）以降低均值标准误（≈CV/√n）；
3. 结论回填 EXP-004 并更新 `docs/research/conclusions.md`。

复现教程:
训练：`SEED=0 sbatch test_dl/train_baseline.slurm`（同理 123）；
汇总：`tools/aggregate_multiseed.py` → `docs/experiments/EXP-005-simvp-multiseed/`。
<!-- INDEX:END -->

<!-- EXP006:BEGIN -->
# EXP-006 抗模糊方案对照（GAN vs 多步训练 / scheduled sampling）

日期：
2026-10-12

实验目的：
长时效预测"越推越模糊"（baseline 在 LT=18 的锐度比仅 0.273）。
本实验对照两类改进方向：① GAN（判别器逼迫输出更真实，生成式，会引入纹理）；
② 多步训练 ± scheduled sampling（纯监督，减少曝光偏差，不引入纹理）。

数据：
- 评估：test 全量 1024 条，滚动 18 步（LT=1..18），逐样本 RMSE(K) 与锐度比
- 训练（GAN 臂）：train 集 35595 条（仅 1 帧真值）
- 训练（多步臂）：validation 集 800 条（有 18 帧真值，可支撑多步监督）

方法：
- **baseline**：单步 MSE（teacher forcing），100 epoch
- **GAN (K=6)**：多步滚动 + 判别器对抗损失（相对权重 5%）+ 双向锐度正则
- **multi-step (p=0)**：多步 MSE，滚 6 步，teacher forcing
- **multi-step + SS (p=0.5)**：多步 MSE + scheduled sampling 概率 0.5

参数：
- 各臂**热启动**于同一个 baseline `best.ckpt`，梯度步数预算均为 **8000 步**
- GAN：rollout K=6、gan_weight=0.05（相对 MSE）、grad_weight=0.05、判别器带谱归一化
- 多步：rollout K=6、lr=5e-4、20 轮 × 400 步

结果：

| 臂 | LT=1 RMSE(K) | LT=18 RMSE(K) | LT=18 锐度比 | 相对 baseline(LT18) |
|---|---|---|---|---|
| baseline | **1.576** | **11.659** | 0.273 | —（基准） |
| GAN (K=6) | **1.807** | **15.719** | 2.969 | 恶化 34.8% ❌ |
| multi-step (p=0) | **1.577** | **13.135** | 0.270 | 恶化 12.7% ❌ |
| multi-step + SS (p=0.5) | **1.597** | **9.901** | 0.122 | **改善 15.1%** ✅ |

> 锐度比 = 预测梯度幅值 / 真实梯度幅值：<1 偏模糊，≈1 与真实相当，>1 有高频噪声。

结论：

- **GAN (K=6)**：RMSE 恶化 34.8% ❌；锐度比 2.969（过冲为噪声）
- **multi-step (p=0)**：RMSE 恶化 12.7% ❌；锐度比 0.270
- **multi-step + SS (p=0.5)**：RMSE **改善 15.1%** ✅；锐度比 0.122

- **根本性权衡**：180 分钟后的细节不可预测，MSE 意义下的最优输出就是平滑的后验均值；任何"补细节"的做法（GAN）必然推高 MSE（感知-失真权衡）。

下一步：
1. 若多步方向有效 → 用更大数据量（validation 全 1024 条）重训，与 baseline 公平对比；
2. 若目标是"清晰且真实的预报"本身，应改用**生成式/概率预报**路线（EDM / 扩散，正是本复现项目的论文 CIRA-Diff 的核心主张），并配套集合与感知类评价指标，而非仅看 RMSE；
3. 结论回填 `docs/research/conclusions.md`。

复现教程:
GAN：`SCHED` 无关，`sbatch test_dl/train_gan.slurm`（ROLLOUT=6）；
多步：`SCHED=0.0 / 0.5 sbatch test_dl/train_rollout.slurm`；
对比：`tools/compare_arms.py` → `docs/experiments/EXP-006-anti-blur/`。

> ⚠️ 对照局限：GAN 臂与多步臂**训练数据来源不同**（train 35595 条 vs validation 800 条），本实验比较的是"同热启动起点、同 8000 步预算下哪种改进方向更有效"，非严格同数据对照。
<!-- EXP006:END -->

---

# EXP-007 感知-失真权衡：确定性 vs 生成式（SimVP / UNet / EDM）

**🧭 主线报告（选定方向：真实性）**：`docs/research/forecast-realism-report.md`
—— 把 EXP-004~007 串成一条主线，可直接作为汇报材料。

| 项 | 内容 |
|---|---|
| 实验 ID | `EXP-007` |
| 状态 | ✅ 完成（LT=1 严格有效；LT≥2 的 EDM 仅作锐度下限） |
| 数据 | test 集**同一批 342 条样本**配对比较；EDM 取自 `ly` 原始预测（**只读**） |
| 结果（单步 t+10min） | SimVP MSE **0.00633** / 锐度 0.822｜UNet 0.00762 / 0.802｜EDM 0.00942 / **0.902** |
| 结论 | **MSE 最优的模型不是最清晰的**——EDM 更清晰但 MSE 差 18%~49%，权衡在全时效一致成立 |
| 数据有效性 | EDM 的 `run_official_edm_multigpu.py` 用 Bug 1 模式 → **LT≥2 作废**；LT=1 teacher-forced 有效 |
| 副线 | SimVP 配对检验显著优于 UNet（t 达 −11 ~ −34.7），是强确定性基线 |
| 产出 | `docs/experiments/EXP-007-perception-distortion/`（图 + 数值 + 结论） |
| 复现 | `python tools/compare_edm_sharpness.py` |
