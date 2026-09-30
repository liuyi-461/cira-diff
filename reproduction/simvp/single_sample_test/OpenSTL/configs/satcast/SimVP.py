method = 'SimVP'

# ---- model ----
# 起点参考 configs/sevir/SimVP.py（SEVIR 是临近预报标准 benchmark，
# 数据形态与 GOES ch13 最接近）。gSTA 是 SimVP 的默认 backbone。
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 64
hid_T = 256
N_T = 4
N_S = 2
drop_path = 0.0

# ---- training ----
lr = 1e-3
sched = 'onecycle'
warmup_epoch = 0
batch_size = 1
val_batch_size = 1

# ---- data ----
# in_shape / pre_seq_length / aft_seq_length / metrics 由
# openstl/datasets/dataset_constant.py 的 dataset_parameters['satcast'] 提供，
# 这里只放数据位置和单样本开关。
data_root = '/data1/satcast'
zarr_name = 'edm_GOES_ch13_train_dataset.zarr'
# 单样本过拟合测试：只取训练集里的第 N 条样本，train/val/test 都用它。
# 设为 None 则是完整数据集训练。
single_sample_idx = 0
num_workers = 2
