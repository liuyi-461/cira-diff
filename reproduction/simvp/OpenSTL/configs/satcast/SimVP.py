# SimVP 全量训练配置 —— GOES-16 ABI ch13，2 帧输入 -> 1 帧输出（单步）
#
# 这是 baseline 对齐版（配合 tools/train_simvp_satcast.py 使用）：
#   * 数据来自三份独立 zarr（train / validation / test），不再是同一份 zarr 抽三个 split；
#   * 不再做子集 z-score —— zarr 里已是归一化空间（mean=0, std=1），模型直接吃原值，
#     因此按本配置训练出来的 checkpoint，评估时用 `--norm-mean 0 --norm-std 1`；
#   * 物理常数 MEAN_K / STD_K 由 dataloader_satcast 提供，指标可还原到亮温(K)。
#
# 重要：本文件是**唯一超参来源**。
#   上游 update_config（openstl/utils/main_utils.py:146）的行为是：
#   "argparse 默认值为真值的 key，配置文件里的值会被静默丢弃"。
#   因此 train_simvp_satcast.py 把本文件的所有 key 都放进 exclude_keys，
#   保证下面的 batch_size / num_workers / lr / epoch 真的生效。
#   代价是这些 key 无法再用命令行覆盖（命令行只保留路径类与冒烟类参数）。
#
# 超参来源说明（取舍记录）：
#   这组数值参考了 cb 副本的 configs/satcast/SimVP.py（cb 已用它跑完 100 epoch 全量训练，
#   得到 test RMSE 1.5745 K）。该结果经我方交叉核对可信（persistence 1024 条复算与 cb
#   偏差 0.0000%），但**这组超参本身并非我方验证过的最优配置**，仅作为"可复现 cb 结果"的
#   一致起点。详见 docs/experiments/EXP-004-simvp-vs-baseline/README.md 的验证记录表。

method = 'SimVP'

# ---- model ----
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 64
hid_T = 256
N_S = 2
N_T = 4
# 全量训练（35595 条、4.7M 参数）开一点正则更稳
drop_path = 0.1

# ---- training ----
# batch_size=8：cb 实测 SimVP 在 256x256 下每样本激活约 2.53 GB，
# batch=8 约 20 GB（48 GB 的 4090 稳妥），batch=16 会 OOM。
# （该实测值为 cb 提供，我方未复测，属参考值。）
batch_size = 8
val_batch_size = 32      # 推理无反向图，显存远低于训练，可开大
num_workers = 8          # 惰性 zarr 读取是主要瓶颈，worker 给足
lr = 2e-3
min_lr = 1e-6
sched = 'onecycle'
warmup_epoch = 0
epoch = 100

# ---- data ----
# 三个 split 各读一份 zarr；validation/test 的 output 有 18 帧（rollout 真值），
# loader 会切出第 0 帧 = t+10min，正是单步任务要的 target。
data_root = '/home/group1/26fall_aiclass/yr/data1/data1/satcast'
train_zarr = 'edm_GOES_ch13_train_dataset.zarr'
val_zarr = 'edm_GOES_ch13_validation_dataset.zarr'
test_zarr = 'edm_GOES_ch13_test_dataset.zarr'

# 注意：刻意**不**在这里写冒烟开关（limit_train 等）。
# 冒烟限样本只在命令行给：--limit-train 64 --limit-val 16 --epochs 1
