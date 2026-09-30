# SimVP 全量训练（GOES-16 ABI ch13，2 帧输入 -> 1 帧输出，单步）

在 CIRA-Diff 原数据上跑 SimVP 的**全量**训练与测试集评估。

> 任务口径（用户 2026-09-28 明确）：**2 帧进、1 帧出**，单步、确定性预报，
> **不做** 18 步 rollout、**不要**集合预报。train 训练 / validation 验证 /
> test 测试，最终报 **RMSE（亮温 K）**。

前序工作见 `../single_sample_test/`（单样本过拟合测试，已跑通，链路结论在那里）。

---

## 一、目录结构

```
full_train/
├── OpenSTL/                  # 复制自 ../single_sample_test/OpenSTL，本目录内做下述修改
├── eval_full_test.py         # 训练后评估：best.ckpt + 全 test 集 + persistence 基线 + RMSE
├── run_full_train.slurm      # slurm 提交脚本（训练 + 评估）
├── logs/
│   ├── output/               # slurm stdout
│   └── err/                  # slurm stderr
├── outputs/                  # 训练产物（checkpoints / 指标 JSON / 对比图 / npy）
└── README.md
```

**为什么从 `single_sample_test/OpenSTL` 复制而不是从 `../../OpenSTL` 本体**：
副本里已经含有单样本测试期间验证过、且与"能否跑起来"直接相关的补丁
（timm 导入回退、satcast 注册、环境修补）。从它复制可以避免把已知能跑通的东西
重新踩一遍坑。`../../OpenSTL` 本体自始至终未被改动。

---

## 二、数据契约

三份 zarr，均在 `/data1/satcast/`，三个 split **各读各的**：

| split | 文件 | input_images | output_images | 本任务取 |
| --- | --- | --- | --- | --- |
| train | `edm_GOES_ch13_train_dataset.zarr` | (35595, 2, 256, 256) | (35595, **1**, 256, 256) | 第 0 帧 |
| val | `edm_GOES_ch13_validation_dataset.zarr` | (1024, 2, 256, 256) | (1024, **18**, 256, 256) | 第 0 帧 |
| test | `edm_GOES_ch13_test_dataset.zarr` | (1024, 2, 256, 256) | (1024, **18**, 256, 256) | 第 0 帧 |

`validation` / `test` 的 `output_images` 是 **18 帧**（Chase 2025 用来做 3 小时
rollout 评测的真值序列），本任务只要单步。**不是随便切第 0 帧**：已验证

```
validation:  |output[:, j] - input[:, 1]|   j = 0..17
   0.078 0.111 0.132 0.151 0.170 0.188 0.206 0.225 0.246 0.265 ... 0.351
   ↑ 随 j 严格单调递增
```

即 `output[:, 0]` 就是 **t+10min**，正是单步任务要的 target。`test` 集同样验证过。

归一化（三个 split 共用训练集统计量）：

```
normalized = (Tb - 279.0699458792467) / 19.32967519050003
```

`dataset.mean/std` 存的是**物理常数**，OpenSTL 的 `metric()` 用它们把预测和真值
反归一化回亮温(K)，所以框架日志里的数字是 K 空间。

张量形状：`(B, T, C, H, W) = (B, 2, 1, 256, 256)`，输出 `(B, 1, 1, 256, 256)`。
zarr 里的 "2" 是**时间维**不是通道维，C=1（单通道灰度亮温）。

---

## 三、相对上游 OpenSTL 的改动

全部在本目录的 `OpenSTL/` 副本内，`../../OpenSTL` 本体保持原样。

| # | 文件 | 改动 | 原因 |
| --- | --- | --- | --- |
| 1 | `openstl/datasets/dataloader_satcast.py` | **重写**：三份 zarr 分 split 加载；`output_images` 沿帧轴切 `[0:aft_seq_length]` | 原来三 split 共用训练集 zarr（单样本测试的有意设计）；且 val/test 的 18 帧形状会被自检直接拒绝 |
| 2 | `openstl/datasets/dataloader.py` | `cfg_dataloader` 白名单加 `val_zarr_name` / `test_zarr_name` | 该 dict 是白名单，不在其中的 kwarg 会被静默丢弃 |
| 3 | `openstl/datasets/dataset_constant.py` | `satcast` 的 `metrics` 加 `'rmse'` | 用户要求最终输出 RMSE |
| 4 | `openstl/core/metrics.py` | `MSE/MAE/RMSE` 的 `.sum()` 改 `.mean()` | **见下方专门说明** |
| 5 | `tools/train.py` | `exclude_keys` 增加 `batch_size`/`data_root`/`num_workers`/`min_lr`/`opt`；新增 `torch.set_float32_matmul_precision('high')` | **见下方专门说明** |
| 6 | `openstl/api/exp.py` | ① `Trainer(..., precision=...)` 把 `--fp16` 接到 Lightning AMP；② checkpoint callback 加 `enable_version_counter=False`；③ `Trainer(..., default_root_dir=save_dir)` 让 `lightning_logs/` 落到 `outputs/<run>/` 而不是代码目录 | **见下方说明 6 / 6b / 6c** |
| 7 | `openstl/utils/parser.py` | 新增 `--single_sample_idx`（默认 None） | 让全链路冒烟测试能从命令行开关 |
| 8 | `configs/satcast/SimVP.py` | **重写** | 全量训练超参 |

### 说明 4：metrics 的 `.sum()` —— 上游有意的约定，不是 bug

上游三个函数都是 `np.mean(..., axis=(0,1)).sum()` —— 在 (B, T) 上取均值、
在 (C, H, W) 上**求和**。即函数名叫 `MSE`，返回的却是空间求和。

**这是上游有意的设计**，维护者 `chengtan9907`（仓库作者）在 issue #134 里明确回复
"在计算metric的时候是对每一帧所有像素求和的"；且已核实上游 `master` 分支至今仍是
`.sum()`（本地副本非篡改）。

影响：同一数据集内倍率 `C·H·W` 恒定，**方法排名不受影响**；但绝对数值无法与其他
代码库比较 —— satcast 场景下 `mse`/`mae` 偏大 **65536 倍**、
`rmse` 偏大 **256 倍**（倍数不同，最容易看漏）。

本任务需要与 CIRA-Diff 在 K 空间逐像素 RMSE 直接可比，所以改为 `.mean()`。
`spatial_norm=True` 分支不在此任务路径上，保持上游原样。

### 说明 5：`update_config` 的"真值陷阱"

`openstl/utils/main_utils.py:141-152` 的逻辑是：**若某 key 的 argparse 默认值为真值
且不在 `exclude_keys` 里，配置文件中的值会被静默丢弃**。上游的 exclude_keys 只有
4 项，于是下面这些写在 `configs/satcast/SimVP.py` 里**不会生效**：

| key | argparse 默认 | 后果 |
| --- | --- | --- |
| `batch_size` | 16 | 配置文件的 8 被丢弃 |
| `data_root` | `'./data'` | 配置文件的 `/data1/satcast` 被丢弃 |
| `num_workers` | 4 | 配置文件的 8 被丢弃 |
| `min_lr` | 1e-6 | 配置值被丢弃 |
| `opt` | `'adam'` | 配置值被丢弃 |

单样本测试是靠"再在命令行传一遍"绕过去的。本次为了让实验配置**只有一个来源**，
把这几项加进 `exclude_keys`（语义变为"一律以配置文件为准"）。**新增配置项时要注意
这个陷阱**：真值默认的 key 必须加进 `exclude_keys`，或走 `dataset_parameters`。

同理新增了 `torch.set_float32_matmul_precision('high')`：torch 2.6 默认
`cudnn.allow_tf32=True`（卷积已用 TF32）但 `float32_matmul_precision='highest'`
（matmul 没用 TF32），而 gSTA 的 MetaBlock MLP 正是 matmul 密集。

### 说明 6：`--fp16` 在上游是**死参数**

上游 `exp.py` 的 `Trainer(...)` 没有传 `precision=`，`args.fp16` 只流到
`datasets/utils.py` 的 prefetcher（那条路径要开 `--use_prefetcher`，默认关）。
即 `--fp16` 对训练**完全无效**。本次把它接到 Lightning 原生 AMP 上。

**默认关闭**（`precision='32-true'`）。要启用必须显式传 `--fp16`，且**建议先跑
冒烟测试确认 loss 不发散**再用于长任务 —— 这是会改变数值行为的改动，尚未在
本模型上实测过。

### 说明 6b：`enable_version_counter=False` —— 续跑会静默丢进度的坑

Lightning 的 `ModelCheckpoint`（`model_checkpoint.py:684-688`）默认行为是：
**若目标文件名已存在，就不覆盖，改写成带版本号的名字**。后果非常隐蔽：

| 步骤 | `last.ckpt` 实际内容 |
| --- | --- |
| 第 1 次跑（干净目录） | 正常，每 epoch 原地更新 ✓ |
| 被 kill，第 2 次 `RESUME=1` 续跑 | 续跑**成功**，但 `last.ckpt` **冻结在上次的位置**，新进度写进 `last-v1.ckpt` ✗ |
| 又被 kill，第 3 次续跑 | 从**冻结的旧位置**恢复，中间进度全丢 ✗ |

**全程不报错、静默发生。** 已通过 `enable_version_counter=False` 修掉，
`last.ckpt` 现在每 epoch 原地覆盖。

> 这个 bug 是实测发现的，不是推理出来的：冒烟测试里 job 61 的 `last.ckpt`
> 停在 17:43，而 job 63 明明训练到 epoch 125，进度全在 `last-v1.ckpt`。
> 若不修，正式任务在第二次续跑时会把第一个 `--time` 窗口内的进度全部丢掉。

### 说明 6c：`default_root_dir=save_dir`

上游没设它，Lightning 的默认 logger 会把 `lightning_logs/` 写到**当前工作目录**
（即 `OpenSTL/` 里面），与"产物都放 `outputs/`"的约定不符。指到 `save_dir` 之后，
它落在 `outputs/<run_name>/lightning_logs/`。

---

## 四、超参（单卡 RTX 4090 48GB）

| 参数 | 值 | 依据 |
| --- | --- | --- |
| `batch_size` | **8** | 显存实测，见下表。非拍脑袋 |
| `val_batch_size` | 32 | 推理无反向图，显存远低于训练 |
| `num_workers` | 8 | 惰性 zarr 读取是主要瓶颈；节点有 64 CPU |
| `epoch` | 100 | 4450 步/epoch → 445,000 步 |
| `lr` | 2e-3 | **最不确定的一项**，见下 |
| `sched` | `onecycle` | 与 `configs/sevir/SimVP.py` 一致 |
| `warmup_epoch` | 0 | onecycle 自身有低起点，与 sevir 一致 |
| `min_lr` | 1e-6 | 默认 |
| `opt` | `adam` | 默认 |
| `drop_path` | 0.1 | 正则；与 sevir 一致（单样本测试时是 0.0） |
| `model_type` | `gSTA` | 默认 backbone，已在单样本测试中跑通 |
| `hid_S / hid_T / N_S / N_T` | 64 / 256 / 2 / 4 | **沿用**单样本配置，不改结构 |

### batch_size 的显存依据

用 `torch.autograd.graph.saved_tensors_hooks` 实测前向保存的激活张量
（按 storage 去重，消除残差连接的重复计数），模型为上述配置：

| batch | 激活(去重) | 每样本 | 含 1.5× 开销估计 | 48GB 判定 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 2.54 GB | 2.544 GB | 3.8 GB | 安全 |
| 4 | 10.12 GB | 2.530 GB | 15.2 GB | 安全 |
| **8** | **20.22 GB** | **2.528 GB** | **30.4 GB** | **安全** |
| 16 | 40.43 GB | 2.527 GB | 60.7 GB | **OOM** |
| 32 | 80.83 GB | 2.526 GB | 121.3 GB | **OOM** |

**每样本约 2.53 GB**，严格线性。这比"4.7M 参数的小模型"直觉上大得多，原因是
SimVP 的解码器在 256×256 全分辨率上工作、且要处理 T=2 帧，加上
`act_inplace` 被上游硬编码为 `False`（激活不就地复用）。

> **注意**：×1.5 是 cuDNN workspace + 显存碎片化的经验估计，不是实测。
> 若 batch=8 仍 OOM（意料之外），降到 4。反之若要压榨速度，12 值得一试
> （约 30 GB 激活，理论可行但余量较小）。

### 速度（实测，2026-09-28 job 74）

| 指标 | 实测值 |
| --- | --- |
| 步速 | **4.27 it/s**（batch=8，即约 34 样本/秒） |
| **每 epoch** | **约 17.4 分钟**（4450 步） |
| **100 epoch 总计** | **约 29 小时** |
| GPU 利用率 | **100%**（采样 5 次稳定在 98–100%） |
| 显存占用 | **21.2 GB**（与下表 batch=8 的预测 ~20 GB 吻合） |
| 功耗 | 442 W |

换算下来有效吞吐约 8.6 TFLOPS（按 84.16 GFLOPs/样本 × 3 计），**比事先估算的
18 TFLOPS 低不少**，所以早先"约 14 小时"的估计偏乐观，实际约 **29 小时**。

**已确认不是 I/O 瓶颈**：GPU 利用率稳定 100%、功耗接近 TDP，dataloader
（第三节提到的 zarr 随机读放大风险）没有拖后腿。29h 是这个模型在 4090 上的真实速度。

> 这直接决定了 `--time` 要设 **48:00:00**：按 24h 设会在 epoch ~83 处被 SLURM 杀掉。

想再提速的话，`--fp16` 是唯一没试过的大杠杆（见第三节说明 6），但需先验证数值稳定性。

---

## 五、运行

```bash
cd /home/group1/26fall_aiclass/cb/cira-diff/reproduction/simvp/full_train
mkdir -p logs/output logs/err        # SLURM 不会自动创建父目录
sbatch run_full_train.slurm

squeue -u $USER
tail -f logs/output/simvp_full_<jobid>.out
```

### 建议流程

**第 1 步：先跑冒烟测试**（约 1 分钟），把 `run_full_train.slurm` 里的 `SMOKE=0`
改成 `SMOKE=1` 再 `sbatch`。它会用**真正的三 split 代码路径**各取 1 条样本跑
2 个 epoch，产物写到 `outputs/satcast_simvp_full_smoke/`，不污染正式结果。
确认数据通路和代码改动都正常后再改回 `SMOKE=0`。

**第 2 步：正式训练。** 超参只需改 `OpenSTL/configs/satcast/SimVP.py` 一处。

**断点续跑**：`sbatch --export=ALL,RESUME=1 run_full_train.slurm` 会从
`checkpoints/last.ckpt` 接着训（Lightning 恢复 epoch / optimizer / scheduler）。
已实测验证：中途 `scancel` 后在 epoch 251 处中断，续跑从 251 之后继续，不报错。

> **改超参后必须让 `RESUME=0`（或删掉 `checkpoints/`）。**
> 原因：`OneCycleLR` 的 `total_steps = epoch × steps_per_epoch` 在创建时就固定。
> 若续跑时改了 `epoch`，恢复出来的调度器会是一个"总长按旧配置"的调度器，
> 步进越界后直接报
> `ValueError: Tried to step N times. The specified number of total steps is M`。
> 这个失败是**立刻发生**的（不浪费训练时间），但会让人困惑。
> 改了 `batch_size` 或数据集规模同理（`steps_per_epoch` 变了）。

### slurm 资源

| 项 | 单样本测试 | 本次 | 说明 |
| --- | --- | --- | --- |
| `partition` | debug | debug（未改） | 已确认该分区 `MaxTime=UNLIMITED` |
| `gres` | gpu:1 | gpu:1（未改） | 节点有 8× RTX 4090 48GB |
| `cpus-per-task` | 4 | **12** | 配合 `num_workers=8` |
| `mem` | 32G | **64G** | 8 个 worker + pin_memory；节点共 514GB |
| `time` | 02:00:00 | **48:00:00** | 100 epoch **实测约 29h**（见第四节速度表）；分区时限 unlimited，给足余量 |

---

## 六、评估

训练结束后 slurm 会自动调用 `eval_full_test.py`：加载 `best.ckpt`，
在**完整 1024 条 test 集**上推理，输出归一化空间与亮温(K)空间两套 MSE/MAE/RMSE，
并与 persistence 基线（把输入最近一帧 t 直接当作 t+10min）对比。

产物：`outputs/satcast_simvp_full/{test_metrics.json, test_result.png, test_preds.npy, test_trues.npy}`

**为什么不直接用 OpenSTL 的 `trainer.test()`**：`tools/train.py:38` 无条件调
`exp.test()`，但 `exp.py:96-99` 只在 `--test` 为真时才加载 `best.ckpt`，
而 `--test` 默认 `False` —— 即框架日志里的 test 数字跑在**最终权重**上。
且 `--test` 会把日志前缀从 `train` 改成 `test`，误导训练记录。
所以权威数字以 `eval_full_test.py` 为准。

（副本里 `openstl/core/metrics.py` 已改为逐像素口径，因此框架日志的 test 数字
与 `eval_full_test.py` 应当一致，可作交叉验证 —— 只是一次在最终权重、
一次在 best.ckpt。）

---

## 七、冒烟测试结果（2026-09-28，SLURM job 66）

`SMOKE=1` 跑通，**exit code 0，无 Traceback**。验证到的内容：

| 检查项 | 结果 |
| --- | --- |
| 三份 zarr 分别加载 | train 35595 / val 1024 / test 1024 ✓ |
| val/test 的 18 帧正确切到 1 帧 | `y == output[:, 0]` 且 `y != output[:, 1]` ✓ |
| **三个 split 确实是不同数据** | val loss(1.77) ≠ train loss(0.52) ✓（旧的单样本设计里两者会是同一条） |
| 超参从配置文件正确生效 | 见 `outputs/.../model_param.json`：batch=8, lr=0.002, epoch, drop_path=0.1, 三份 zarr 名 ✓ |
| metrics 补丁生效 | 框架日志输出逐像素口径，不再是像素和 ✓ |
| 评估脚本跑完整 test 集 | 1024 条，产出 JSON / PNG / npy ✓ |
| 断点续跑 | epoch 251 中断 → `RESUME=1` 从 251 之后继续 ✓ |
| checkpoint 命名 | 无 `last-vN.ckpt`，`last.ckpt` 为最新 ✓ |

**注意冒烟测试的指标本身没有意义**（模型只在 3 条样本上训练了 2 个 epoch），
日志里 `RMSE(K)=16.65` 远差于 persistence 的 `4.55`、预测范围出现 `-139.7 ~ 354.6 K`
这种非物理值是**预期现象**。它证明的是链路通，不是模型好。

---

## 八、已知限制 / 注意事项

- **`lr = 2e-3` 是最不确定的超参。** 上游 SEVIR 配置是 `lr=5e-3, batch=8`，
  weather 是 `lr=1e-3`，单样本测试用了 `lr=1e-3, batch=1`。本任务取中间值。
  这是没有经过验证的选择，若有异常（loss 震荡/不收敛）优先调它。
- **跑到一半被 kill**：`--time` 到期后 SLURM 会终止任务，用 `RESUME=1` 续跑。
- **`epoch=100` 是否够**：未知。SimVP 在 OpenSTL 的默认是 200 epoch，
  但那是针对小数据集；本任务数据量大得多。看 val_loss 曲线判断。
- **dataloader 的 I/O 放大**：train zarr 的 chunk 是 `(45, 2, 256, 256)` ≈ 11.8 MB，
  而一条样本只有约 267 KB，随机读取理论上约 46 倍放大。节点有 514 GB 内存，
  页缓存可以吸收大部分，但**未实测**。若训练明显 I/O 瓶颈，考虑全量缓存进 RAM
  （约 13 GiB，需调大 `--mem`）。
- **`fp16` 未实测**，见第三节说明 6。
- 本 README 中所有时间/速度数字都是**估算**，显存数字是 CPU 侧实测激活量 + 经验开销系数。
