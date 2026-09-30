# 训练配置（Training）

## 默认 gSTA 配置
来源：`configs/mmnist/simvp/SimVP_gSTA.py`
```
method='SimVP'
spatio_kernel_enc=3
spatio_kernel_dec=3
model_type='gSTA'
hid_S=64
hid_T=512
N_T=8
N_S=4
lr=1e-3
batch_size=16
drop_path=0
sched='onecycle'
```

## 运行命令
```shell
python tools/train.py -d mmnist --lr 1e-3 -c configs/mmnist/simvp/SimVP_gSTA.py --ex_name mmnist_simvp_gsta
```

## 配置优先级
- `--overwrite`：用 config 覆盖 CLI 默认（排除 `method`）。
- 否则：用 config 更新 CLI 默认（排除 `method, val_batch_size, drop_path, warmup_epoch`），
  缺失项回填 `default_parser()`。
