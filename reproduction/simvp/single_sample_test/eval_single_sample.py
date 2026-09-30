#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SimVP 单样本测试的**训练后评估**：加载 checkpoint，对那一条样本做推理，
并与 persistence 基线对比。

为什么要单独写这个脚本：OpenSTL 的 trainer.test() 只输出 mse/mae 两个数字，
没有对照。单样本测试是"过拟合/记忆"测试，一个没有基线的 MSE 数字说明不了
任何事情（模型可能什么都没学到、输出一张糊图，MSE 依然"看起来还行"）。
这里补上 persistence 基线（直接把最近的输入帧 t 当作 t+10min 的预测），
和 cira-diff 的 train_vanilla_unet_Chase2025.py 里的做法一致。

指标同时给归一化空间和亮温(K)空间，方便与 /data1/satcast 那边的记录对齐。

用法（需要 cira-diff-cb 环境 + PYTHONPATH 指向 OpenSTL 目录）：
    PYTHONPATH=<OpenSTL> python eval_single_sample.py \
        --ckpt <...>/checkpoints/best.ckpt --idx 0
"""

import argparse
import json
import math
import os
import os.path as osp
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

HERE = osp.dirname(osp.abspath(__file__))
OPENSTL_DIR = osp.join(HERE, 'OpenSTL')
if OPENSTL_DIR not in sys.path:
    sys.path.insert(0, OPENSTL_DIR)

from openstl.datasets import dataset_parameters                       # noqa: E402
from openstl.datasets.dataloader_satcast import (MEAN_K, STD_K,       # noqa: E402
                                                 SatcastZarrDataset)
from openstl.models import SimVP_Model                                # noqa: E402
from openstl.utils import load_config                                 # noqa: E402

# SimVP_Model.__init__ 接受的键（config 文件里还有 lr/epoch 等训练超参，要过滤掉）
MODEL_KEYS = ('in_shape', 'hid_S', 'hid_T', 'N_S', 'N_T', 'model_type', 'mlp_ratio',
              'drop', 'drop_path', 'spatio_kernel_enc', 'spatio_kernel_dec',
              'act_inplace')


def to_K(x):
    """归一化空间 -> 亮温(K)。"""
    return x * STD_K + MEAN_K


def build_model(config_path):
    """按 configs/satcast/SimVP.py + dataset_parameters['satcast'] 构建模型。"""
    cfg = dict(load_config(config_path))
    cfg.update(dataset_parameters['satcast'])          # in_shape 等
    kwargs = {k: cfg[k] for k in MODEL_KEYS if k in cfg}
    return SimVP_Model(**kwargs), kwargs


def load_weights(model, ckpt_path):
    """把 Lightning checkpoint 里的权重装进裸 SimVP_Model。

    checkpoint 的 state_dict 键带 'model.' 前缀（SimVP 这个 LightningModule
    把模型存在 self.model 里），这里剥掉前缀；同时校验确实全部匹配上了，
    免得"加载成功"其实只是随机初始化。
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    sd = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd.items()}

    model_keys = set(model.state_dict().keys())
    missing = model_keys - set(sd.keys())
    unexpected = set(sd.keys()) - model_keys
    if missing or unexpected:
        raise RuntimeError(
            f'权重与模型不匹配：missing={sorted(missing)[:5]} '
            f'unexpected={sorted(unexpected)[:5]}'
        )
    model.load_state_dict(sd, strict=True)
    return ckpt


def evaluate(ckpt_path, config_path, data_root, zarr_name, idx, out_dir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model, kwargs = build_model(config_path)
    ckpt = load_weights(model, ckpt_path)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print('=' * 72)
    print('SimVP 单样本评估')
    print('=' * 72)
    print(f'[ENV]   device     : {device}')
    print(f'[CKPT]  {ckpt_path}')
    print(f'[CKPT]  epoch      : {ckpt.get("epoch", "N/A")}'
          f' （共 {ckpt.get("global_step", "N/A")} 个全局步）')
    print(f'[MODEL] {kwargs}')
    print(f'[MODEL] 参数量 {n_params / 1e6:.3f} M')

    # ---- 数据：与训练时完全相同的那一条样本 ----
    zarr_path = zarr_name if osp.isabs(zarr_name) else osp.join(data_root, zarr_name)
    dataset = SatcastZarrDataset(zarr_path, pre_seq_length=2, aft_seq_length=1)
    x, y = dataset[idx]                      # (2,1,256,256), (1,1,256,256)
    x_in = x.unsqueeze(0).to(device)         # (1,2,1,256,256)
    y_gt = y.unsqueeze(0).to(device)         # (1,1,1,256,256)
    print(f'[DATA]  {zarr_path}  第 {idx} 条（数据集共 {len(dataset)} 条）')
    print(f'[DATA]  x={tuple(x_in.shape)}  y={tuple(y_gt.shape)}')

    with torch.no_grad():
        # 复现 openstl/methods/simvp.py 中 aft_seq_length < pre_seq_length 的路径：
        # 模型输出 T=2 帧，取前 1 帧作为预测。
        pred_full = model(x_in)
        pred = pred_full[:, :1]
    pred = pred.reshape_as(y_gt)             # (1,1,1,256,256) -> 与 y_gt 同形

    # ---- persistence 基线：把最近一帧输入 t 直接当作 t+10min ----
    persistence = x_in[:, -1:].reshape_as(y_gt)

    def metrics(p, t):
        mse = F.mse_loss(p, t).item()
        mae = F.l1_loss(p, t).item()
        return mse, mae, math.sqrt(mse) * STD_K

    mse, mae, rmse_K = metrics(pred, y_gt)
    mse_p, mae_p, rmse_p_K = metrics(persistence, y_gt)

    # 逐通道指标（本任务 C=1，等价于整体）
    reduction = (1.0 - mse / mse_p) * 100.0 if mse_p > 0 else float('nan')

    print('\n' + '-' * 72)
    print(f'{"指标":<24}{"SimVP 预测":>16}{"persistence 基线":>20}')
    print('-' * 72)
    print(f'{"MSE  (归一化空间)":<24}{mse:>16.6f}{mse_p:>20.6f}')
    print(f'{"MAE  (归一化空间)":<24}{mae:>16.6f}{mae_p:>20.6f}')
    print(f'{"RMSE (亮温 K)":<24}{rmse_K:>16.4f}{rmse_p_K:>20.4f}')
    print('-' * 72)
    print(f'MSE 相对 persistence 基线降低：{reduction:.2f}%')
    print(f'\n预测范围 (K) : {to_K(pred).min().item():.1f} ~ {to_K(pred).max().item():.1f}')
    print(f'真值范围 (K) : {to_K(y_gt).min().item():.1f} ~ {to_K(y_gt).max().item():.1f}')
    print(f'输入 t   (K) : {to_K(x_in[:, -1:]).min().item():.1f} ~ '
          f'{to_K(x_in[:, -1:]).max().item():.1f}')

    # ---- 出图 ----
    out_png = osp.join(out_dir, 'single_sample_result.png')
    try:
        panels = (
            (x_in[0, 0, 0].cpu().numpy(), 'Input t-10min', 'Spectral_r', -4, 2),
            (x_in[0, 1, 0].cpu().numpy(), 'Input t', 'Spectral_r', -4, 2),
            (y_gt[0, 0, 0].cpu().numpy(), 'Target t+10min (GT)', 'Spectral_r', -4, 2),
            (pred[0, 0, 0].cpu().numpy(), f'SimVP pred (MSE={mse:.4f})', 'Spectral_r', -4, 2),
            (persistence[0, 0, 0].cpu().numpy(),
             f'Persistence (MSE={mse_p:.4f})', 'Spectral_r', -4, 2),
            (np.abs(pred[0, 0, 0].cpu().numpy() - y_gt[0, 0, 0].cpu().numpy()),
             f'|SimVP - GT| (MAE={mae:.4f})', 'hot', None, None),
        )
        fig, axes = plt.subplots(1, 6, figsize=(27, 4.6))
        for ax, (img, title, cmap, vmin, vmax) in zip(axes, panels):
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.suptitle(
            f'SimVP single-sample test | 2 frames in -> 1 frame out | '
            f'RMSE {rmse_K:.3f} K vs persistence {rmse_p_K:.3f} K', fontsize=12)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'\n[SAVE] 对比图：{out_png}')
    except Exception as exc:
        print(f'\n[WARNING] 出图失败（不影响指标）：{type(exc).__name__}: {exc}')

    # ---- 落盘指标，便于别人/后续引用 ----
    result = {
        'ckpt': ckpt_path,
        'ckpt_epoch': ckpt.get('epoch', None),
        'sample_index': idx,
        'zarr': zarr_path,
        'params_M': round(n_params / 1e6, 4),
        'simvp': {'mse_norm': mse, 'mae_norm': mae, 'rmse_K': rmse_K},
        'persistence': {'mse_norm': mse_p, 'mae_norm': mae_p, 'rmse_K': rmse_p_K},
        'mse_reduction_vs_persistence_pct': reduction,
    }
    out_json = osp.join(out_dir, 'single_sample_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'[SAVE] 指标 JSON：{out_json}')

    print('\n[注意] 单样本过拟合测试：只说明整条链路跑通、且模型学下了这一条样本，')
    print('       不代表任何泛化能力。')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=osp.join(
        HERE, 'outputs', 'satcast_simvp_single_sample', 'checkpoints', 'best.ckpt'))
    ap.add_argument('--config', default=osp.join(
        OPENSTL_DIR, 'configs', 'satcast', 'SimVP.py'))
    ap.add_argument('--data_root', default='/data1/satcast')
    ap.add_argument('--zarr_name', default='edm_GOES_ch13_train_dataset.zarr')
    ap.add_argument('--idx', type=int, default=0)
    ap.add_argument('--out_dir', default=osp.join(
        HERE, 'outputs', 'satcast_simvp_single_sample'))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not osp.exists(args.ckpt):
        raise FileNotFoundError(f'找不到 checkpoint：{args.ckpt}')
    evaluate(args.ckpt, args.config, args.data_root, args.zarr_name, args.idx, args.out_dir)


if __name__ == '__main__':
    main()
