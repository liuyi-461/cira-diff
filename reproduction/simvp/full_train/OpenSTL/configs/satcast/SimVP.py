# SimVP 全量训练配置 —— GOES-16 ABI ch13，2 帧输入 -> 1 帧输出（单步）
#
# 目标硬件：单卡 RTX 4090 48GB。
#
# 重要：本文件是**唯一**的超参来源。为了让下面的 batch_size / num_workers /
# data_root / min_lr 真的生效，本副本的 tools/train.py 已经把这几项加进了
# update_config 的 exclude_keys（上游行为是：argparse 默认值为真值的 key，
# 配置文件里的值会被静默丢弃）。详见该文件内的注释。
#
# 数据契约由 openstl/datasets/dataset_constant.py 的 dataset_parameters['satcast']
# 提供：in_shape=[2,1,256,256]、pre_seq_length=2、aft_seq_length=1、metrics。

method = 'SimVP'

# ---- model ----
# 起点参考 configs/sevir/SimVP.py（SEVIR 是临近预报标准 benchmark，数据形态与
# GOES ch13 最接近）。gSTA 是 SimVP 的默认 backbone；这组尺寸已在单样本测试
# 中确认可跑通，全量训练沿用，不改结构。
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 64
hid_T = 256
N_T = 4
N_S = 2
# 单样本测试时是 0.0。全量训练有 35595 条样本、4.7M 参数，开一点正则更稳。
drop_path = 0.1

# ---- training ----
# batch_size 由实测显存决定，不是拍的：SimVP 在 256x256 下**每个样本的激活约
# 2.53 GB**（实测，见 README 的表），batch=16 需要约 40 GB 激活 + 开销，
# 在 48 GB 的 4090 上会 OOM；batch=8 约 20 GB，是稳的。
# 这一取值也和上游 configs/sevir/SimVP.py 的 batch_size 一致。
batch_size = 8
val_batch_size = 32      # 推理无反向图，显存远低于训练，可以开大
num_workers = 8          # 惰性 zarr 读取是主要瓶颈，worker 给足
lr = 2e-3                # 最不确定的一项，见 README
min_lr = 1e-6
sched = 'onecycle'       # 与 configs/sevir/SimVP.py 一致
warmup_epoch = 0         # onecycle 自身有低起点，与 sevir 配置一致
epoch = 100              # onecycle 的调度长度与训练长度同源，改这里两处一起变

# ---- data ----
# 三个 split 各读一份 zarr（用户 2026-09-28 指定）。
# validation / test 的 output_images 是 18 帧 rollout 真值，loader 会切出第 0 帧
# = t+10min，正是单步任务要的 target。
data_root = '/data1/satcast'
zarr_name = 'edm_GOES_ch13_train_dataset.zarr'
val_zarr_name = 'edm_GOES_ch13_validation_dataset.zarr'
test_zarr_name = 'edm_GOES_ch13_test_dataset.zarr'

# 全链路冒烟测试开关在**命令行**，不在这里：
#     --single_sample_idx 0
# 效果：三个 split 各取自己 zarr 的第 0 条、batch_size 强制为 1，
# 用来在提交长任务前确认三份数据通路都通。正式训练不要传这个参数。
#
# 注意：刻意**不**把 single_sample_idx 写进本文件。update_config 的逻辑是
# "配置文件里存在的 key，若命令行值为假值（0/None）则由配置文件覆盖"，
# 写在这里会让 --single_sample_idx 0 失效。
