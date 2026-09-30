#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SimVP 全量训练后的**测试集评估**：加载 best.ckpt，在完整 test zarr 上推理，
输出逐像素 RMSE（亮温 K），并与 persistence 基线对比。

为什么单独写这个脚本（而不是直接用 OpenSTL 的 trainer.test()）：

  1. OpenSTL 的 test 默认跑在**最终权重**上，不是 best.ckpt
     （tools/train.py:38 无条件调 exp.test()，但 exp.py 只在 --test 为真时加载
     best.ckpt，而 --test 默认 False）。
  2. 单步任务的 RMSE 需要一个对照基线才说明得了问题。这里补上 persistence
     （把输入最近一帧 t 直接当作 t+10min），与 cira-diff 侧的惯例一致。
  3. 本脚本同时给出**归一化空间**和**亮温(K)空间**两套数字，方便和
     /data1/satcast 那边的记录对齐。

注意：本仓库副本里 openstl/core/metrics.py 的 MSE/MAE/RMSE 已被改成逐像素均值
（上游是空间求和）。本脚本**不依赖**那个函数，是自己算的，口径为
     MSE = 全部元素的均方误差；RMSE = sqrt(MSE)
两者应当一致，可以互相交叉验证。

用法（需要 cira-diff-cb 环境；PYTHONPATH 指向本目录的 OpenSTL）：
    PYTHONPATH=<full_train>/OpenSTL python eval_full_test.py \
        --ckpt <...>/checkpoints/best.ckpt --split test
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

HERE = osp.dirname(osp.abspath(__file__))
OPENSTL_DIR = osp.join(HERE, 'OpenSTL')
if OPENSTL_DIR not in sys.path:
    sys.path.insert(0, OPENSTL_DIR)

from openstl.datasets import dataset_parameters                        # noqa: E402
from openstl.datasets.dataloader_satcast import (MEAN_K, STD_K,        # noqa: E402
                                                 SatcastZarrDataset)
from openstl.models import SimVP_Model                                 # noqa: E402
from openstl.utils import load_config                                  # noqa: E402

# SimVP_Model.__init__ 接受的键（config 里还有 lr/epoch 等训练超参，要过滤掉）
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

    checkpoint 的 state_dict 键带 'model.' 前缀（SimVP 这个 LightningModule 把
    模型存在 self.model 里），这里剥掉前缀；同时校验全部匹配上了，免得
    "加载成功"其实只是随机初始化。
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


@torch.no_grad()
def run_inference(model, dataset, device, batch_size):
    """在整个数据集上分批推理，返回 (pred, true, persist, inp_last) —— 都是 (N,1,1,256,256)。"""
    preds, trues, persists, lasts = [], [], [], []
    model.eval()
    n = len(dataset)
    for start in range(0, n, batch_size):
        idxs = list(range(start, min(start + batch_size, n)))
        xb = torch.stack([dataset[i][0] for i in idxs]).to(device)   # (b,2,1,256,256)
        yb = torch.stack([dataset[i][1] for i in idxs]).to(device)   # (b,1,1,256,256)

        # 复现 openstl/methods/simvp.py 中 aft_seq_length < pre_seq_length 的路径：
        # 模型输出 T=2 帧，取前 1 帧作为预测。
        pred = model(xb)[:, :1]

        preds.append(pred.cpu())
        trues.append(yb.cpu())
        # persistence 基线：把最近的输入帧 t 直接当作 t+10min
        persists.append(xb[:, -1:].cpu())
        lasts.append(xb[:, -1:].cpu())

    cat = lambda xs: torch.cat(xs, dim=0).numpy().astype(np.float64)  # noqa: E731
    return cat(preds), cat(trues), cat(persists), cat(lasts)


def metrics(p, t):
    """逐像素口径：MSE = 全部元素均方误差，RMSE = sqrt(MSE)，单位随输入。"""
    mse = float(((p - t) ** 2).mean())
    mae = float(np.abs(p - t).mean())
    return mse, mae, math.sqrt(mse)


def save_figure(out_png, pred, true, persist, n_show=4):
    """挑前 n_show 条测试样本画对比图。"""
    n_show = min(n_show, pred.shape[0])
    fig, axes = plt.subplots(n_show, 5, figsize=(22, 4.2 * n_show))
    axes = np.atleast_2d(axes)
    for r in range(n_show):
        gt = true[r, 0, 0]
        pd = pred[r, 0, 0]
        ps = persist[r, 0, 0]
        panels = (
            (gt, f'#{r} Target t+10min (GT)', 'Spectral_r', -4, 2),
            (pd, f'#{r} SimVP pred', 'Spectral_r', -4, 2),
            (ps, f'#{r} Persistence', 'Spectral_r', -4, 2),
            (np.abs(pd - gt), f'#{r} |SimVP - GT|', 'hot', None, None),
            (np.abs(ps - gt), f'#{r} |Persistence - GT|', 'hot', None, None),
        )
        for ax, (img, title, cmap, vmin, vmax) in zip(axes[r], panels):
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)


def evaluate(ckpt_path, config_path, data_root, zarr_name, out_dir, device_str,
             batch_size, n_show):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    model, kwargs = build_model(config_path)
    ckpt = load_weights(model, ckpt_path)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    zarr_path = zarr_name if osp.isabs(zarr_name) else osp.join(data_root, zarr_name)
    dataset = SatcastZarrDataset(zarr_path, pre_seq_length=2, aft_seq_length=1)

    print('=' * 74)
    print('SimVP 测试集评估（2 帧 -> 1 帧，单步）')
    print('=' * 74)
    print(f'[ENV]   device  : {device}')
    print(f'[CKPT]  {ckpt_path}')
    print(f'[CKPT]  epoch   : {ckpt.get("epoch", "N/A")}')
    print(f'[MODEL] {kwargs}')
    print(f'[MODEL] 参数量 {n_params / 1e6:.3f} M')
    print(f'[DATA]  {zarr_path}')
    print(f'[DATA]  {len(dataset)} 条测试样本（output 原始 {dataset.out_frames} 帧，'
          f'本任务取第 0 帧 = t+10min）')
    print(f'[RUN]   batch_size={batch_size}')

    pred, true, persist, _ = run_inference(model, dataset, device, batch_size)
    assert pred.shape == true.shape, (pred.shape, true.shape)

    mse, mae, rmse = metrics(pred, true)
    mse_p, mae_p, rmse_p = metrics(persist, true)

    reduction = (1.0 - mse / mse_p) * 100.0 if mse_p > 0 else float('nan')

    print('\n' + '-' * 74)
    print(f'{"指标":<26}{"SimVP 预测":>16}{"persistence":>18}')
    print('-' * 74)
    print(f'{"MSE   (归一化空间)":<26}{mse:>16.6f}{mse_p:>18.6f}')
    print(f'{"MAE   (归一化空间)":<26}{mae:>16.6f}{mae_p:>18.6f}')
    print(f'{"RMSE  (归一化空间)":<26}{rmse:>16.6f}{rmse_p:>18.6f}')
    print('-' * 74)
    print(f'{"MSE   (K^2)":<26}{mse * STD_K**2:>16.4f}{mse_p * STD_K**2:>18.4f}')
    print(f'{"MAE   (K)":<26}{mae * STD_K:>16.4f}{mae_p * STD_K:>18.4f}')
    print(f'{"RMSE  (K)  <-- 主指标":<26}{rmse * STD_K:>16.4f}{rmse_p * STD_K:>18.4f}')
    print('-' * 74)
    print(f'MSE 相对 persistence 降低：{reduction:.2f}%')

    # 逐样本 RMSE(K)，看分布而不是只看均值
    per_sample = np.sqrt(((pred - true) ** 2).mean(axis=(1, 2, 3, 4))) * STD_K
    print(f'\n逐样本 RMSE(K): 均值 {per_sample.mean():.4f} | 中位数 '
          f'{np.median(per_sample):.4f} | 最差 {per_sample.max():.4f} | '
          f'最好 {per_sample.min():.4f}')

    # 越界检查
    print(f'预测范围 (K): {to_K(pred).min():.1f} ~ {to_K(pred).max():.1f}')
    print(f'真值范围 (K): {to_K(true).min():.1f} ~ {to_K(true).max():.1f}')

    os.makedirs(out_dir, exist_ok=True)
    out_png = osp.join(out_dir, 'test_result.png')
    try:
        save_figure(out_png, pred, true, persist, n_show=n_show)
        print(f'\n[SAVE] 对比图   : {out_png}')
    except Exception as exc:
        print(f'\n[WARNING] 出图失败（不影响指标）：{type(exc).__name__}: {exc}')

    result = {
        'ckpt': ckpt_path,
        'ckpt_epoch': ckpt.get('epoch', None),
        'zarr': zarr_path,
        'n_samples': int(len(dataset)),
        'params_M': round(n_params / 1e6, 4),
        'simvp': {
            'mse_norm': mse, 'mae_norm': mae, 'rmse_norm': rmse,
            'mse_K2': mse * STD_K**2, 'mae_K': mae * STD_K, 'rmse_K': rmse * STD_K,
        },
        'persistence': {
            'mse_norm': mse_p, 'mae_norm': mae_p, 'rmse_norm': rmse_p,
            'mse_K2': mse_p * STD_K**2, 'mae_K': mae_p * STD_K,
            'rmse_K': rmse_p * STD_K,
        },
        'per_sample_rmse_K': {
            'mean': float(per_sample.mean()), 'median': float(np.median(per_sample)),
            'max': float(per_sample.max()), 'min': float(per_sample.min()),
        },
        'mse_reduction_vs_persistence_pct': reduction,
    }
    out_json = osp.join(out_dir, 'test_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'[SAVE] 指标 JSON: {out_json}')

    np.save(osp.join(out_dir, 'test_preds.npy'), pred.astype(np.float32))
    np.save(osp.join(out_dir, 'test_trues.npy'), true.astype(np.float32))
    print(f'[SAVE] 预测/真值 .npy 已存到 {out_dir}')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=osp.join(
        HERE, 'outputs', 'satcast_simvp_full', 'checkpoints', 'best.ckpt'))
    ap.add_argument('--config', default=osp.join(
        OPENSTL_DIR, 'configs', 'satcast', 'SimVP.py'))
    ap.add_argument('--data_root', default='/data1/satcast')
    ap.add_argument('--zarr_name', default='edm_GOES_ch13_test_dataset.zarr',
                    help='评估用哪份 zarr；默认 test 集')
    ap.add_argument('--out_dir', default=osp.join(
        HERE, 'outputs', 'satcast_simvp_full'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--n_show', type=int, default=4)
    args = ap.parse_args()

    if not osp.exists(args.ckpt):
        raise FileNotFoundError(f'找不到 checkpoint：{args.ckpt}')
    evaluate(args.ckpt, args.config, args.data_root, args.zarr_name,
             args.out_dir, args.device, args.batch_size, args.n_show)


if __name__ == '__main__':
    main()
