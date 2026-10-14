# tools/ — 训练、评估、对比与报告脚本

> 本目录是本复现项目的**全部自研逻辑**。上游 OpenSTL（`openstl/models/`、
> `openstl/core/` 等）未做任何修改，只新增了 `openstl/datasets/dataloader_satcast.py`
> 这一个数据适配器（见文末「上游边界」）。
>
> 所有脚本都可以**不依赖外部 PYTHONPATH** 直接运行（内部已做 `sys.path` 自举）。

## 环境

```bash
conda activate openstl          # 克隆自 test 环境 + timm==0.9.12 等依赖
```

⚠️ 不要用 `test` 环境：会在 `import openstl` 阶段因 `timm>=1.0` 报错（见 `docs/dev/debugging.md` DBG-005）。

## 脚本总览

| 分组 | 脚本 |
|---|---|
| **数据与训练** | `train_simvp_satcast.py`、`train_simvp_gan.py`、`train_simvp_rollout.py` |
| **评估** | `eval_simvp_baseline.py` |
| **对比分析** | `compare_with_unet_raw.py`、`plot_unet_comparison.py`、`aggregate_multiseed.py`、`compare_arms.py`、`compare_edm_sharpness.py`、`error_buckets.py` |
| **报告与导出** | `make_report_figures.py`、`finalize_baseline_report.py`、`export_preds_official_format.py` |
| **索引同步** | `sync_research_index.py`、`sync_exp006_index.py` |
| （历史） | `train_simvp_goes13_smoke.py`、`train_simvp_goes13_single_sample.py`（EXP-002/003 用） |

配套提交脚本在 `test_dl/*.slurm`，均为「环境自检 + 绝对路径 + GPU 独占」结构。

---

## 一、训练

### 1.1 `train_simvp_satcast.py` — baseline（确定性 SimVP 全量训练）

**用途**：按 baseline 协议训练 SimVP（三份独立 zarr、不做子集 z-score、物理常数归一化）。

```bash
python tools/train_simvp_satcast.py \
    --config configs/satcast/SimVP.py \
    --res_dir /abs/path/OpenSTL/work_dirs \
    --ex_name satcast_simvp_full
```

**参数**
| 参数 | 说明 |
|---|---|
| `--config` | 超参唯一来源（默认 `configs/satcast/SimVP.py`） |
| `--res_dir` / `--ex_name` | 产物落点 `<res_dir>/<ex_name>/`（**建议绝对路径**） |
| `--epochs` | 覆盖 config 的 epoch 数 |
| `--satcast-root` | 覆盖数据根目录 |
| `--limit-train/val/test` | 各 split 只取前 N 条，**仅用于冒烟**，正式训练勿用 |
| `--cpu` / `--run-test` | CPU 运行 / 训完跑测试 |
| `--seed`、`--ckpt_path`、`--dist`、`--use-prefetcher` | 来自上游 `create_parser()`，`--ckpt_path` 用于续跑 |

**产出**：`work_dirs/<ex_name>/{checkpoints/best.ckpt, last.ckpt, model_param.json}`

> ⚠️ `--ex_name` 不要留默认值 `Debug`（脚本会警告）；`--res_dir` 会被自动解析为绝对路径（DBG-004 护栏）。
> ⚠️ 配置文件里的超参**必须**在 `exclude_keys` 中才会生效——脚本已把全部 key 放入，并有一条自检：
> 实际值与期望值逐项比对，见脚本内 `expect` 字典。

### 1.2 `train_simvp_gan.py` — GAN 抗模糊训练

**用途**：训练时滚动 K 步，把滚动结果送判别器，用对抗损失逼迫输出更真实。

```bash
python tools/train_simvp_gan.py \
    --ex-name satcast_simvp_gan --rollout-steps 6 \
    --gan-weight 0.05 --grad-weight 0.05 \
    --init-ckpt work_dirs/satcast_simvp_full/checkpoints/best.ckpt
```

**关键参数**：`--rollout-steps K`（训练时滚几步）、`--gan-weight`（对抗损失，**相对 MSE 的比例**）、
`--grad-weight`（双向锐度正则）、`--detach-steps`（只对最后一步回传，省显存）、
`--lr-d`（判别器学习率）、`--init-ckpt`（热启动，强烈推荐）

**损失形式**：`G = mse × (1 + gan_w·adv + grad_w·rel_grad)`
> ⚠️ **量纲是关键**：mse≈6.6e-3 而 adv/rg 都是 O(1)，相差约 75 倍。
> 若写成 `mse + w·adv`，即使 w=0.01 也会让对抗项与 mse 同量级并带偏模型
> （实测导致 LT=18 RMSE 恶化 85.8%）。权重必须定义为「相对 mse 的比例」。

**实测结论**：GAN 在所有时效上都**未能降低 RMSE**（LT=18 恶化 34.8%），且锐度比冲到 2.97（噪声）。
这是感知-失真权衡的必然，非调参问题——详见 `docs/research/forecast-realism-report.md`。

### 1.3 `train_simvp_rollout.py` — 多步训练 + scheduled sampling

**用途**：对全部 K 步逐一算 MSE，并按概率 p 喂模型自己的预测以减少曝光偏差。
**这是本次唯一被证实有效的确定性改良**。

```bash
python tools/train_simvp_rollout.py \
    --ex-name satcast_simvp_rollout_ss05_full \
    --rollout-steps 6 --sched-samp 0.5 \
    --n-train 1024 --n-val 512 --val-source train \
    --epochs 16 --init-ckpt work_dirs/satcast_simvp_full/checkpoints/best.ckpt
```

**关键参数**
| 参数 | 说明 |
|---|---|
| `--rollout-steps K` | 滚动步数 |
| `--sched-samp p` | scheduled sampling 概率（0=纯 teacher forcing） |
| `--val-source validation` | 从多步训练集中切验证集（训练集上限 1024−n_val 条） |
| `--val-source train` | 验证改用 train zarr 留出子集 → **训练集可用满全量** |
| `--n-train` / `--n-val` | 训练/验证样本数 |
| `--init-ckpt` | 热启动（推荐 baseline best.ckpt） |

**实测结论**（3 seed，全量 1024 条）：LT=18 RMSE **9.904 ± 0.121 K**，相对 baseline **改善 15.1% ± 1.0 pp**
（CV 仅 1.22%，12.3σ 显著）；但**锐度比同时从 0.273 降到 0.133**（减半）。

> ⚠️ 数据来源：train 集只有 1 帧真值，**无法**做多步监督；多步训练必须用 validation 集
> （其 `output` 有 18 帧）。test 集始终 held-out。

---

## 二、评估

### `eval_simvp_baseline.py` — 统一协议评估

**用途**：加载 best.ckpt，按 baseline 协议评估（单步 + 可选 18 步 rollout），
逐像素口径，同时报归一化值与**亮温 K**，并附 persistence 基线。

```bash
# 单步（test 全量 1024 条）
python tools/eval_simvp_baseline.py --ckpt <...>/best.ckpt \
    --norm-mean 0 --norm-std 1 --out_dir <...>/baseline_eval

# 18 步 rollout（默认自动取 128 条子集，与 ly 的 UNet 协议一致）
python tools/eval_simvp_baseline.py --ckpt <...>/best.ckpt \
    --rollout-steps 18 --out_dir <...>/baseline_eval_rollout18

# 全量 1024 条 rollout（配对检验用，功效更高）
python tools/eval_simvp_baseline.py --ckpt <...>/best.ckpt \
    --rollout-steps 18 --max-samples 1024 --out_dir <...>/baseline_eval_rollout18_full
```

**关键参数**：`--rollout-steps`（1=单步；18=3 小时）、`--max-samples`（0/不设=自动：rollout 取 128、单步取全量）、
`--norm-mean/--norm-std`（还原训练时的归一化；**新训练用 `0/1`**，EXP-002 老 ckpt 用 `0.093006/0.865920`）

**产出**：`baseline_metrics.json`、`baseline_curves.png`、`baseline_panels.png`、`preds/trues/persist.npy`

---

## 三、对比分析

### `compare_with_unet_raw.py` — SimVP vs UNet 配对对比

读 `ly` 的 UNet 原始预测数组，做**同批样本的配对检验**，并出并排云图+残差。

```bash
python tools/compare_with_unet_raw.py --n-samples 1024 \
    --pred-subdir baseline_eval_rollout18_full
```

- `--pred-subdir`：`baseline_eval_rollout18`（128 条）或 `baseline_eval_rollout18_full`（1024 条）
- 自动校验真值逐位对齐；同时算**公平口径**（两边非集成）与**集成口径**（两边都集成）

**产出**：`unet_vs_simvp_clouds.png`、`unet_vs_simvp_paired.png`、`unet_vs_simvp_paired.json`，
并**自动回填** EXP-005 README 的 PAIRED 标记段。

### `plot_unet_comparison.py` — 多 seed 逐时效曲线

SimVP(多 seed) vs UNet(ly, 3 seed) 的 MSE / RMSE(K) / SSIM 曲线，含 ±1 std 带与 persistence。
→ `unet_vs_simvp_rollout.png` / `.csv`

### `aggregate_multiseed.py` — 多 seed 汇总

逐 LT 的 mean±std、CV、胜负判定、稳健性结论；同时落 `multiseed_summary.json` 供索引同步。
→ EXP-005 README 的 MULTISEED 标记段

### `compare_arms.py` — 七臂统一对比

baseline / GAN / 多步(p=0) / 多步+SS / 全量SS / 全量SS-seed1 / 全量SS-seed7 的
逐时效 RMSE 与锐度比 + 自动判定 + 图。
→ `fig_anti_blur_compare.png` + EXP-006 README

### `compare_edm_sharpness.py` — 感知-失真权衡实测

SimVP / UNet / EDM 在**同一批 342 条样本**上的单步 MSE 与锐度比。
EDM 数据取自 `ly` 的原始预测（**只读**），**仅用 LT=1**（其 rollout 有 Bug 1，LT≥2 作废）。
→ `fig_perception_distortion.png` + EXP-007

### `error_buckets.py` — 分组诊断

按**云量 / 对流强度 / 场景变化强度**分桶，看 RMSE 在各类场景的表现。
→ `fig_error_buckets.png`

> ⚠️ 「按时段分桶做不了」：test zarr **不含时间戳**（上游 patch 随机抽取）。

---

## 四、报告与导出

### `make_report_figures.py` — 复刻 cb 的分析图

- `fig_training_curve.png`：训练曲线，纵轴换算为**亮温 K 的 RMSE**，含 SLURM 重启标注
- `fig_error_analysis.png`：(a) 逐样本 RMSE 分布 (b) 误差随目标亮温 (c)(d) 空间平均有符号偏差

### `export_preds_official_format.py` — 官方格式预测导出

导出 `(1024, 18, 256, 256)` float32（样本, lead time, H, W），与 `ly` 的 `pred_official.npy` 同格式。

```bash
python tools/export_preds_official_format.py \
    --extra gan=satcast_simvp_gan rollout_ss05_full_seed1=satcast_simvp_rollout_ss05_full_seed1
```

→ `work_dirs/satcast_preds/{pred_simvp_seed*.npy, pred_gan.npy, pred_rollout_*.npy, pred_simvp_ens.npy, gt.npy}`

### `finalize_baseline_report.py` — 自动生成 EXP-004 Results 段

读评估 JSON，渲染 markdown 并替换 EXP-004 README 的 `RESULTS:BEGIN/END` 标记区（幂等）。

### `sync_research_index.py` / `sync_exp006_index.py` — 索引同步

把 EXP-004/005 与 EXP-006 的结论同步进 `docs/research/experiments.md`（各自带标记区，幂等）。

---

## 五、常用配方

```bash
# 1) 从头训 baseline（100 epoch）
sbatch test_dl/train_baseline.slurm

# 2) 多 seed（与 ly 的 seed 集合对齐）
SEED=0 sbatch test_dl/train_baseline.slurm
SEED=123 sbatch test_dl/train_baseline.slurm

# 3) 训完评估（单步 + 18 步 rollout）
CKPT=work_dirs/satcast_simvp_full/checkpoints/best.ckpt \
  NORM_MEAN=0 NORM_STD=1 ROLLOUT=18 sbatch test_dl/eval_baseline.slurm

# 4) 全量 1024 条 rollout（配对检验用）
python tools/eval_simvp_baseline.py --ckpt <ckpt> --rollout-steps 18 \
    --max-samples 1024 --norm-mean 0 --norm-std 1 --out_dir <dir>

# 5) scheduled sampling 全量重训（推荐配置）
SCHED=0.5 ROLLOUT=6 EPOCHS=16 N_TRAIN=1024 N_VAL=512 VAL_SOURCE=train \
  EX_NAME=satcast_simvp_rollout_ss05_full sbatch test_dl/train_rollout.slurm

# 6) 一键：评估 + 导出 + 七臂对比
sbatch test_dl/eval_rollout_arms.slurm

# 7) 重算所有报告图表
python tools/make_report_figures.py && python tools/error_buckets.py \
  && python tools/plot_unet_comparison.py \
  && python tools/compare_with_unet_raw.py --n-samples 1024 \
       --pred-subdir baseline_eval_rollout18_full \
  && python tools/compare_arms.py && python tools/compare_edm_sharpness.py
```

---

## 六、踩过的坑（改代码前务必先看）

| # | 坑 | 规避方式 |
|---|---|---|
| 1 | **上游 `update_config` 会静默丢弃配置值**（argparse 默认值为真值的 key） | 所有配置 key 必须进 `exclude_keys`；`train_simvp_satcast.py` 内有自检 |
| 2 | **`res_dir`/`ex_name` 默认是相对路径** → 产物落点随 cwd 漂移（DBG-004） | 脚本自动转绝对路径并警告；提交时传绝对路径 |
| 3 | **量纲陷阱**：GAN 对抗损失 O(1) vs MSE O(1e-3) | 权重定义为「相对 MSE 的比例」 |
| 4 | **`torch.where` 广播坑**：条件张量少一维时，batch 维会被广播到时间轴 | 条件张量必须与被选张量**同维数**；`rollout_with_ss` 内有逐步断言 |
| 5 | **SimVP 输出多一维**：`model(x)[:, :1]` 仍是 `(B,1,C,H,W)` | 收集结果时 `squeeze(1)`；滑动窗口用带时间维的那份 |
| 6 | **`nvidia-smi ... \| head -1`** 在 `set -o pipefail` 下因 SIGPIPE 直接杀脚本 | 加 `\|\| true` |
| 7 | **OneCycleLR 的 `total_steps` 不能拍脑袋** | 必须用 `len(train_loader) × epochs` |
| 8 | **`.sum()` 口径 bug**（上游 `metrics.py` 把 MSE 放大 65536 倍） | 不要用 `exp.test()` 的数字；一律用 `eval_simvp_baseline.py` 的逐像素口径 |
| 9 | **rollout 评估默认只取 128 条** | 配对检验要 `--max-samples 1024`，否则标准误偏大 |

---

## 七、上游边界（遵守 DEC-003）

- `openstl/models/`、`openstl/core/`、`openstl/utils/` 等**未做任何修改**。
- 唯一新增：`openstl/datasets/dataloader_satcast.py`（satcast 专用适配器）。
- 所有自研逻辑都在 `tools/` 内，通过 `BaseExperiment(dataloaders=...)` / 独立训练循环注入。

---

## 八、可移植性（给别人用之前必读）

### 8.1 三步复用

```bash
# ① 拷贝这四样（缺一不可）
cp -r tools/ test_dl/ configs/satcast/ <对方OpenSTL>/
cp openstl/datasets/dataloader_satcast.py <对方OpenSTL>/openstl/datasets/

# ② 建/激活环境（清单见 docs/dev/environment.md，timm 必须 <1.0）
conda activate openstl

# ③ 自检依赖
cd <对方OpenSTL> && python tools/check_portability.py
```

`check_portability.py` 会检查 9 项：适配器是否存在、Python 包、配置文件、三份 zarr、
**`update_config` 是否会静默丢弃配置**、ly 产物、GPU、slurm 是否还有个人路径。
**退出码 0 才能提交作业。**

### 8.2 环境变量（替代改代码）

| 变量 | 作用 | 默认值 |
|---|---|---|
| `SATCAST_ROOT` | 数据根目录（三份 zarr 所在） | `/home/group1/26fall_aiclass/yr/data1/data1/satcast` |
| `LY_OUTPUTS` | ly 的产物根（UNet/EDM 原始预测） | `/home/group1/26fall_aiclass/ly/cira-diff/outputs` |
| `OPENSTL_DIR` | 仓库根（slurm 用） | 由脚本自身位置推导 |

```bash
SATCAST_ROOT=/data1/satcast python tools/train_simvp_satcast.py ...
LY_OUTPUTS=/path/to/ly/outputs python tools/compare_with_unet_raw.py ...
```

> ⚠️ 本集群上有**两份内容相同但 inode 不同**的 satcast 副本：
> `/data1/satcast/`（ly 惯用）与 `/home/group1/26fall_aiclass/yr/data1/data1/satcast/`（本项目默认）。
> 二者内容已用 ly 的 `gt.npy` 与我方 `trues.npy` **逐位比对一致**（差 `0.0`），
> 但默认值指向后者——**别人用时请显式设 `SATCAST_ROOT`**，以免误读他人副本。

### 8.3 各类脚本的可移植性分级

| 级别 | 脚本 | 说明 |
|---|---|---|
| ✅ 直接可用 | `train_simvp_satcast.py`、`eval_simvp_baseline.py`、`make_report_figures.py`、`error_buckets.py`、`check_portability.py` | 路径全有 CLI / 环境变量 |
| ✅ 可用（需指定 `--init-ckpt`） | `train_simvp_gan.py`、`train_simvp_rollout.py` | 热启动权重需自己提供；不提供则从头训 |
| ⚠️ 需 `LY_OUTPUTS` | `compare_with_unet_raw.py`、`compare_edm_sharpness.py`、`export_preds_official_format.py` | 读 ly 的原始预测数组；对 ly 本人开箱即用 |
| ⚠️ 需自己的产物 | `finalize_baseline_report.py`、`sync_research_index.py`、`aggregate_multiseed.py`、`compare_arms.py`、`plot_unet_comparison.py` | 默认指向本项目 `work_dirs`，可用 CLI 覆盖 |

### 8.4 🔇 最危险的一项：配置被静默丢弃

`train_simvp_satcast.py` 靠把配置 key 全塞进 `exclude_keys`，绕过上游 `update_config`
的丢弃逻辑（argparse 默认值为真值的 key 会被静默覆盖）。

**如果对方的 OpenSTL 版本与本站不同**，配置会被静默丢弃 → 用 argparse 默认值训练，
**不报错、不警告，但模型超参完全不对**。

规避：
1. `python tools/check_portability.py` 里的「配置生效自检」会实跑一次解析，
   逐项比对 12 个关键超参，不一致即报**阻断**；
2. 训练脚本启动时也会打印配置生效情况。

### 8.5 已消除的硬编码（2026-10-14）

| 原问题 | 现状 |
|---|---|
| 16 个 slurm 硬编码 `OPENSTL_DIR=/home/group1/...` | 改为从脚本自身位置推导：`"${OPENSTL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"` |
| 16 个 slurm 硬编码 `#SBATCH --output=/home/...` | 改为相对提交目录 `test_dl/logs/xxx_%j.out` |
| 9 个 slurm 硬编码 `DATA_ROOT=/home/...` | 改为 `"${SATCAST_ROOT:-<默认>}"` |
| 2 个 slurm 的 `INIT_CKPT` 指向我的 best.ckpt | 改为 `$OPENSTL_DIR/work_dirs/...`（相对仓库根） |
| 7 个脚本硬编码数据根 / ly 产物路径 | 全部改为环境变量优先（见 8.2） |
